"""
URDF 加载与关节驱动模块

核心原理：
Rerun 的变换累乘依赖实体路径的层级关系。
必须用嵌套路径反映 URDF 的父子关系，例如：

  hand/left/left_hand_palm_link                                          (根，Identity)
  hand/left/left_hand_palm_link/left_hand_thumb_0_link                   (相对 palm 的局部变换)
  hand/left/left_hand_palm_link/left_hand_thumb_0_link/left_hand_thumb_1_link  (相对 thumb_0 的局部变换)
  ...

这样 Rerun 才会沿路径层级自动累乘变换，手指节段才能正确首尾相连。

每帧只需更新有关节角度的 link 的 Transform3D（局部变换 = origin_t + origin_R @ R_joint）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False
    print("[警告] rerun-sdk 未安装，请运行: pip install rerun-sdk")

try:
    import yourdfpy
    HAS_YOURDFPY = True
except ImportError:
    HAS_YOURDFPY = False

from .config_loader import HandConfig, JointInfo


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 旋转公式：绕 axis 旋转 angle 弧度，返回 3x3 旋转矩阵"""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _mat3_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → 四元数 [x, y, z, w]（Rerun 格式）"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x = 0.25 / s, (R[2,1] - R[1,2]) * s
        y, z = (R[0,2] - R[2,0]) * s, (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x = (R[2,1] - R[1,2]) / s, 0.25 * s
        y, z = (R[0,1] + R[1,0]) / s, (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x = (R[0,2] - R[2,0]) / s, (R[0,1] + R[1,0]) / s
        y, z = 0.25 * s, (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x = (R[1,0] - R[0,1]) / s, (R[0,2] + R[2,0]) / s
        y, z = (R[1,2] + R[2,1]) / s, 0.25 * s
    return np.array([x, y, z, w])


class UrdfHandLoader:
    """
    负责将一个灵巧手 URDF 加载到 Rerun，并驱动关节动画。

    使用方式:
        loader = UrdfHandLoader(config, side="left")
        loader.log_static(rec)                              # 加载静态 mesh
        loader.log_frame(rec, joint_angles, frame_index=0) # 写入第 t 帧
    """

    def __init__(self, config: HandConfig, side: str = "left", entity_root: str = "hand"):
        self.config = config
        self.side = side
        self.entity_root = entity_root
        self.joint_infos: List[JointInfo] = sorted(
            config.joints.get(side) or config.joints.get("default", []),
            key=lambda j: j.dof_index
        )
        self.joint_names = [j.name for j in self.joint_infos]

        # 从 yourdfpy 解析的关节信息
        self._urdf_model = None
        self._joint_axes: Dict[str, np.ndarray] = {}       # joint_name -> axis (3,)
        self._joint_origin_t: Dict[str, np.ndarray] = {}   # joint_name -> translation (3,)
        self._joint_origin_R: Dict[str, np.ndarray] = {}   # joint_name -> rotation 3x3
        self._joint_child: Dict[str, str] = {}             # joint_name -> child_link_name
        self._joint_parent: Dict[str, str] = {}            # joint_name -> parent_link_name
        self._fixed_joints: set = set()                    # fixed 关节名集合
        self._prismatic_joints: set = set()                # prismatic（平移）关节名集合
        self._root_link: Optional[str] = None              # 根连杆名（palm）

        # link_name -> 嵌套实体路径（含 entity_root 前缀）
        # 例如: "left_hand_thumb_1_link" ->
        #   "hand/left/left_hand_palm_link/left_hand_thumb_0_link/left_hand_thumb_1_link"
        self._link_entity_paths: Dict[str, str] = {}

        # link_name -> STL 文件路径
        self._link_stl_paths: Dict[str, Optional[Path]] = {}
        # link_name -> mesh scale (3,)，默认 [1,1,1]
        self._link_mesh_scale: Dict[str, np.ndarray] = {}
        # link_name -> visual origin 旋转矩阵（mesh 在 link 坐标系里的额外旋转）
        self._link_visual_R: Dict[str, np.ndarray] = {}
        # link_name -> visual origin 平移（mesh 在 link 坐标系里的额外平移）
        self._link_visual_t: Dict[str, np.ndarray] = {}

    def _load_urdf_model(self) -> bool:
        """用 yourdfpy 解析 URDF，构建关节信息和嵌套路径映射"""
        if not HAS_YOURDFPY:
            print("[警告] yourdfpy 未安装")
            return False

        urdf_path = self.config.get_urdf_path(self.side)
        if urdf_path is None or not urdf_path.exists():
            return False

        try:
            # load_meshes=False 跳过 trimesh 解析（GLB 格式会报错），只需关节结构
            self._urdf_model = yourdfpy.URDF.load(str(urdf_path), load_meshes=False)

            for jname, j in self._urdf_model.joint_map.items():
                # 旋转轴
                self._joint_axes[jname] = np.array(
                    j.axis if j.axis is not None else [0, 0, 1], dtype=float
                )
                # origin（yourdfpy 0.0.60 返回 4x4 ndarray）
                if j.origin is not None and isinstance(j.origin, np.ndarray) and j.origin.shape == (4, 4):
                    self._joint_origin_t[jname] = j.origin[:3, 3].copy()
                    self._joint_origin_R[jname] = j.origin[:3, :3].copy()
                elif j.origin is not None and hasattr(j.origin, 'xyz'):
                    self._joint_origin_t[jname] = np.array(j.origin.xyz)
                    self._joint_origin_R[jname] = np.eye(3)
                else:
                    self._joint_origin_t[jname] = np.zeros(3)
                    self._joint_origin_R[jname] = np.eye(3)

                self._joint_child[jname] = j.child
                self._joint_parent[jname] = j.parent
                if j.type == "fixed":
                    self._fixed_joints.add(jname)
                elif j.type == "prismatic":
                    self._prismatic_joints.add(jname)

            # 找根连杆（没有被任何 joint 作为 child 的 link）
            all_children = set(self._joint_child.values())
            all_links = set(self._urdf_model.link_map.keys())
            roots = all_links - all_children
            self._root_link = next(iter(roots)) if roots else list(all_links)[0]

            # 构建嵌套路径：从根开始 BFS，路径 = 父路径 + "/" + 子link名
            self._link_entity_paths = {}
            self._link_entity_paths[self._root_link] = f"{self.entity_root}/{self._root_link}"

            # child_link -> parent_link 的反向映射
            child_to_parent: Dict[str, str] = {v: k for k, v in self._joint_child.items()}
            # BFS
            queue = [self._root_link]
            while queue:
                current_link = queue.pop(0)
                current_path = self._link_entity_paths[current_link]
                # 找所有以 current_link 为 parent 的 joint
                for jname, parent_link in self._joint_parent.items():
                    if parent_link == current_link:
                        child_link = self._joint_child[jname]
                        self._link_entity_paths[child_link] = f"{current_path}/{child_link}"
                        queue.append(child_link)

            # 构建 mesh 路径映射（支持 STL / GLB / OBJ）
            mesh_search_dirs = self.config.get_mesh_search_dirs(self.side)
            for lname, link in self._urdf_model.link_map.items():
                mesh_path = None
                mesh_scale = np.ones(3)
                visual_R   = np.eye(3)
                # 优先从 URDF visual 标签读取 mesh 文件名
                if (link.visuals and
                        link.visuals[0].geometry is not None and
                        hasattr(link.visuals[0].geometry, 'mesh') and
                        link.visuals[0].geometry.mesh is not None):
                    vis = link.visuals[0]
                    mesh_filename = vis.geometry.mesh.filename
                    # 读取 mesh scale
                    if vis.geometry.mesh.scale is not None:
                        mesh_scale = np.array(vis.geometry.mesh.scale, dtype=float)
                    # 读取 visual origin 旋转 + 平移（mesh 在 link 坐标系里的额外变换）
                    if vis.origin is not None and isinstance(vis.origin, np.ndarray):
                        visual_R = vis.origin[:3, :3].copy()
                    candidate = urdf_path.parent / mesh_filename
                    if candidate.exists():
                        mesh_path = candidate
                    else:
                        # 只取文件名部分，在搜索目录中查找
                        fname = Path(mesh_filename).name
                        for d in mesh_search_dirs:
                            c = d / fname
                            if c.exists():
                                mesh_path = c
                                break
                # 若找到的是 .obj/.STL，检查同目录是否有同名 .glb（优先用 glb，渲染效果更好）
                if mesh_path is not None and mesh_path.suffix.lower() in ('.obj', '.stl'):
                    glb_candidate = mesh_path.with_suffix('.glb')
                    if not glb_candidate.exists():
                        glb_candidate = mesh_path.with_suffix('.GLB')
                    if glb_candidate.exists():
                        mesh_path = glb_candidate
                # 若 URDF 中无 mesh 或找不到，按 link 名在搜索目录中查找（glb 优先）
                if mesh_path is None:
                    for ext in [".glb", ".GLB", ".STL", ".stl", ".obj", ".OBJ"]:
                        for d in mesh_search_dirs:
                            c = d / f"{lname}{ext}"
                            if c.exists():
                                mesh_path = c
                                break
                        if mesh_path is not None:
                            break
                # 读取 visual origin 平移
                visual_t = np.zeros(3)
                if (link.visuals and link.visuals[0].origin is not None
                        and isinstance(link.visuals[0].origin, np.ndarray)):
                    visual_t = link.visuals[0].origin[:3, 3].copy()
                self._link_stl_paths[lname]  = mesh_path
                self._link_mesh_scale[lname] = mesh_scale
                self._link_visual_R[lname]   = visual_R
                self._link_visual_t[lname]   = visual_t

            print(f"[✓] yourdfpy 解析完成：{len(self._joint_child)} 个关节，根连杆={self._root_link}")
            print(f"[✓] 嵌套路径示例:")
            for lname, path in list(self._link_entity_paths.items())[:3]:
                print(f"      {lname} -> {path}")
            return True

        except Exception as e:
            print(f"[警告] yourdfpy 解析失败: {e}")
            import traceback; traceback.print_exc()
            return False

    def log_static(self, rec: "rr.RecordingStream") -> bool:
        """
        用 rr.Asset3D 手动加载每个 STL mesh（静态）。
        实体路径使用嵌套结构，反映 URDF 父子关系。

        关键：对每个 link 都写一个静态初始 Transform3D，包含该 link 对应关节的
        origin 平移 + 旋转。fixed 关节的坐标系旋转必须在这里写入，否则子链方向错误。
        """
        if not HAS_RERUN:
            return False

        urdf_path = self.config.get_urdf_path(self.side)
        if urdf_path is None or not urdf_path.exists():
            print(f"[警告] URDF 文件不存在: {urdf_path}")
            return False

        if not self._load_urdf_model():
            return False

        # 构建 child_link -> joint_name 的映射（用于查 origin）
        child_to_joint: Dict[str, str] = {v: k for k, v in self._joint_child.items()}

        loaded = 0
        for lname, entity_path in self._link_entity_paths.items():
            # mesh 放到 entity_path/mesh 子路径，避免 Transform3D 和 Asset3D 互相干扰
            # 在 /mesh 上写 scale + visual_R（static），mesh 本身不受关节动画影响
            stl_path = self._link_stl_paths.get(lname)
            if stl_path and stl_path.exists():
                mesh_path = f"{entity_path}/mesh"
                scale    = self._link_mesh_scale.get(lname, np.ones(3))
                visual_R = self._link_visual_R.get(lname, np.eye(3))
                visual_t = self._link_visual_t.get(lname, np.zeros(3))
                # scale 统一值（若三轴相同）
                uniform_scale = float(scale[0]) if (np.allclose(scale[0], scale[1]) and np.allclose(scale[0], scale[2])) else 1.0
                # 写 mesh 的局部变换（visual_t + visual_R + scale），static
                rec.log(
                    mesh_path,
                    rr.Transform3D(
                        translation=visual_t.tolist(),
                        rotation=rr.Quaternion(xyzw=_mat3_to_quat(visual_R)),
                        scale=uniform_scale,
                    ),
                    static=True,
                )
                rec.log(mesh_path, rr.Asset3D(path=str(stl_path)), static=True)
                loaded += 1
            else:
                print(f"  [跳过] 找不到 mesh: {lname}")

        print(f"[✓] 已加载 {loaded} 个 mesh → {self.entity_root}")
        return loaded > 0

    def _log_initial_transforms(self, rec: "rr.RecordingStream", timeline: str, frame_index: int) -> None:
        """
        在指定帧写入所有 link 的初始 Transform3D（含 fixed 关节的 origin 旋转）。
        必须走时序（非 static），否则会覆盖后续动态帧。
        """
        child_to_joint: Dict[str, str] = {v: k for k, v in self._joint_child.items()}
        rr.set_time(timeline, sequence=frame_index)

        for lname, entity_path in self._link_entity_paths.items():
            if lname == self._root_link:
                rec.log(entity_path, rr.Transform3D(translation=[0, 0, 0]))
            else:
                jname = child_to_joint.get(lname)
                if jname is not None:
                    t = self._joint_origin_t.get(jname, np.zeros(3))
                    R = self._joint_origin_R.get(jname, np.eye(3))
                    quat = _mat3_to_quat(R)
                    rec.log(
                        entity_path,
                        rr.Transform3D(
                            translation=t.tolist(),
                            rotation=rr.Quaternion(xyzw=quat),
                        ),
                    )

    def _resolve_urdf_joint_name(self, config_name: str) -> Optional[str]:
        """
        将 config 中的关节名映射到 URDF 中实际存在的关节名。
        config 可能带 rh_/lh_/right_/left_ 前缀，URDF 可能没有（或反之）。
        """
        if config_name in self._joint_child:
            return config_name
        # config 带前缀，URDF 不带 -> 去掉前缀再查
        for prefix in ("rh_", "lh_", "right_hand_", "left_hand_", "right_", "left_"):
            if config_name.startswith(prefix):
                stripped = config_name[len(prefix):]
                if stripped in self._joint_child:
                    return stripped
                break
        # URDF 带前缀，config 不带 -> 遍历 URDF 关节名
        for urdf_name in self._joint_child:
            for prefix in ("rh_", "lh_", "right_hand_", "left_hand_", "right_", "left_"):
                if urdf_name.startswith(prefix) and urdf_name[len(prefix):] == config_name:
                    return urdf_name
        return None

    def log_frame(
        self,
        rec: "rr.RecordingStream",
        joint_angles: np.ndarray,
        timestamp: float = 0.0,
        timeline: str = "frame_nr",
        frame_index: int = 0,
    ) -> None:
        """
        向 Rerun 写入一帧：更新每个关节对应 child link 的局部 Transform3D。

        Transform3D 写到嵌套路径，是相对于父 link 的局部变换：
            translation = origin_t
            rotation    = origin_R @ R_joint(angle)

        Rerun 沿路径层级自动累乘，得到正确的世界坐标。
        """
        if not HAS_RERUN:
            return

        rr.set_time(timeline, sequence=frame_index)

        # 每帧都写根连杆和所有 fixed 关节的变换，保证坐标系链路完整
        # （fixed 关节的 origin 旋转必须每帧都在，否则 Rerun 跨帧继承不可靠）
        if self._root_link and self._root_link in self._link_entity_paths:
            rec.log(self._link_entity_paths[self._root_link],
                    rr.Transform3D(translation=[0, 0, 0]))
        for jname in self._fixed_joints:
            child_link = self._joint_child.get(jname)
            entity_path = self._link_entity_paths.get(child_link) if child_link else None
            if entity_path is None:
                continue
            t = self._joint_origin_t.get(jname, np.zeros(3))
            R = self._joint_origin_R.get(jname, np.eye(3))
            rec.log(entity_path, rr.Transform3D(
                translation=t.tolist(),
                rotation=rr.Quaternion(xyzw=_mat3_to_quat(R)),
            ))

        n = min(len(joint_angles), len(self.joint_infos))
        for i in range(n):
            joint_info = self.joint_infos[i]
            # 宽松匹配：config 名可能带前缀，URDF 名可能不带（或反之）
            jname = self._resolve_urdf_joint_name(joint_info.name)
            if jname is None:
                continue
            angle = float(np.clip(joint_angles[i], joint_info.limit_lower, joint_info.limit_upper))

            child_link = self._joint_child.get(jname)
            if child_link is None:
                continue

            entity_path = self._link_entity_paths.get(child_link)
            if entity_path is None:
                continue

            axis = self._joint_axes.get(jname, np.array([0.0, 0.0, 1.0]))
            t_origin = self._joint_origin_t.get(jname, np.zeros(3))
            R_origin = self._joint_origin_R.get(jname, np.eye(3))

            if jname in self._prismatic_joints:
                # prismatic 关节：值是沿 axis 方向的平移量（米）
                # 局部变换：平移 = origin_t + axis * displacement，旋转 = origin_R（不变）
                t_local = t_origin + axis * angle
                rec.log(
                    entity_path,
                    rr.Transform3D(
                        translation=t_local.tolist(),
                        rotation=rr.Quaternion(xyzw=_mat3_to_quat(R_origin)),
                    ),
                )
            else:
                # revolute 关节：值是绕 axis 的旋转角度（弧度）
                # 局部变换：平移 = origin_t，旋转 = origin_R @ R_joint
                R_joint = _rotation_matrix_from_axis_angle(axis, angle)
                R_local = R_origin @ R_joint
                rec.log(
                    entity_path,
                    rr.Transform3D(
                        translation=t_origin.tolist(),
                        rotation=rr.Quaternion(xyzw=_mat3_to_quat(R_local)),
                    ),
                )

    def log_sequence(
        self,
        rec: "rr.RecordingStream",
        joint_angles_seq: np.ndarray,
        fps: float = 30.0,
        timeline: str = "frame_nr",
        start_time: float = 0.0,
    ) -> None:
        """批量写入关节角度序列，shape=(T, dof)"""
        T = joint_angles_seq.shape[0]
        for t in range(T):
            self.log_frame(
                rec, joint_angles_seq[t],
                timestamp=start_time + t / fps,
                timeline=timeline,
                frame_index=t,
            )
        print(f"[✓] 已写入 {T} 帧动画数据（{T/fps:.2f}s @ {fps}fps）")
