"""
HandVisualizer：统一的灵巧手可视化入口

使用示例:
    import numpy as np
    from hand_viz import HandVisualizer

    # 创建可视化器
    viz = HandVisualizer("unitree_dex3")

    # 方式1：可视化单帧
    joint_angles = np.zeros(7)
    viz.visualize_frame(joint_angles, side="left")

    # 方式2：可视化动作序列（T帧 × DOF）
    seq = np.random.uniform(-0.5, 0.5, (100, 7))
    viz.visualize(seq, side="left", fps=30.0)

    # 方式3：同时可视化左右手
    left_seq  = np.random.uniform(-0.5, 0.5, (100, 7))
    right_seq = np.random.uniform(-0.5, 0.5, (100, 7))
    viz.visualize_bimanual(left_seq, right_seq, fps=30.0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

from .config_loader import HandConfig, load_hand_config
from .urdf_loader import UrdfHandLoader


class HandVisualizer:
    """
    灵巧手可视化器。

    支持:
    - 单手/双手可视化
    - 单帧/序列动画
    - 保存为 .rrd 文件或实时流式传输到 Rerun Viewer
    """

    def __init__(
        self,
        hand_name: str,
        app_id: str = "dexterous_hand_viz",
        spawn_viewer: bool = True,
    ):
        """
        Args:
            hand_name: 手型 ID，如 "unitree_dex3"
            app_id: Rerun 应用 ID
            spawn_viewer: 是否自动启动 Rerun Viewer 窗口
        """
        if not HAS_RERUN:
            raise ImportError("请先安装 rerun-sdk: pip install rerun-sdk")

        self.hand_name = hand_name
        self.app_id = app_id
        self.spawn_viewer = spawn_viewer

        # 加载手型配置
        self.config: HandConfig = load_hand_config(hand_name)
        print(f"[HandVisualizer] 已加载配置: {self.config.display_name}（{self.config.dof} DOF）")

        # 初始化 Rerun
        self._rec: Optional[rr.RecordingStream] = None

    def _init_rerun(self, recording_id: Optional[str] = None) -> rr.RecordingStream:
        """初始化或复用 Rerun 录制流"""
        if self._rec is not None:
            return self._rec

        rec = rr.new_recording(
            application_id=self.app_id,
            recording_id=recording_id,
            make_default=True,
        )

        if self.spawn_viewer:
            rr.spawn(memory_limit="500MB")

        # 设置坐标系（机器人惯例：Z 向上）
        rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._rec = rec
        return rec

    def _make_blueprint(self, has_left: bool, has_right: bool) -> "rrb.Blueprint":
        """生成 Rerun 布局蓝图"""
        views = []
        if has_left:
            views.append(
                rrb.Spatial3DView(
                    name="左手",
                    origin="hand/left",
                )
            )
        if has_right:
            views.append(
                rrb.Spatial3DView(
                    name="右手",
                    origin="hand/right",
                )
            )
        if has_left and has_right:
            views.append(
                rrb.Spatial3DView(
                    name="双手总览",
                    origin="hand",
                )
            )

        return rrb.Blueprint(
            rrb.Horizontal(*views) if len(views) > 1 else views[0],
            collapse_panels=True,
        )

    def visualize(
        self,
        joint_angles_seq: np.ndarray,
        side: str = "left",
        fps: float = 30.0,
        save_path: Optional[Union[str, Path]] = None,
        timeline: str = "frame_nr",
    ) -> None:
        """
        可视化关节角度序列（动画）。

        Args:
            joint_angles_seq: shape=(T, dof) 或 (dof,) 单帧
                              单位：弧度
            side: "left" 或 "right"
            fps: 帧率
            save_path: 若指定，保存为 .rrd 文件（如 "output.rrd"）
            timeline: Rerun 时间轴名称
        """
        # 统一为 2D
        angles = np.atleast_2d(joint_angles_seq)
        if angles.shape[0] == 1 and joint_angles_seq.ndim == 1:
            # 单帧输入
            pass

        rec = self._init_rerun()

        # 设置蓝图
        blueprint = self._make_blueprint(
            has_left=(side == "left"),
            has_right=(side == "right"),
        )
        rec.send_blueprint(blueprint)

        # 创建加载器
        entity_root = f"hand/{side}"
        loader = UrdfHandLoader(self.config, side=side, entity_root=entity_root)

        # 加载静态 URDF
        urdf_ok = loader.log_static(rec)
        if not urdf_ok:
            print(f"[提示] URDF 未找到，仅记录关节角度数据（无 3D 模型）")

        # 写入序列
        loader.log_sequence(rec, angles, fps=fps, timeline=timeline)

        # 保存文件
        if save_path is not None:
            save_path = Path(save_path)
            rr.save(str(save_path), default_blueprint=blueprint)
            print(f"[✓] 已保存录制文件: {save_path}")

    def visualize_frame(
        self,
        joint_angles: np.ndarray,
        side: str = "left",
        timestamp: float = 0.0,
    ) -> None:
        """
        可视化单帧关节角度。

        Args:
            joint_angles: shape=(dof,)，单位：弧度
            side: "left" 或 "right"
            timestamp: 时间戳（秒）
        """
        rec = self._init_rerun()
        entity_root = f"hand/{side}"
        loader = UrdfHandLoader(self.config, side=side, entity_root=entity_root)
        loader.log_static(rec)
        loader.log_frame(rec, joint_angles, timestamp=timestamp, frame_index=0)

    def visualize_bimanual(
        self,
        left_seq: np.ndarray,
        right_seq: np.ndarray,
        fps: float = 30.0,
        save_path: Optional[Union[str, Path]] = None,
        timeline: str = "frame_nr",
    ) -> None:
        """
        同时可视化左右手序列。

        Args:
            left_seq:  左手关节角度序列，shape=(T, dof)
            right_seq: 右手关节角度序列，shape=(T, dof)
            fps: 帧率
            save_path: 保存路径（可选）
            timeline: 时间轴名称
        """
        left_seq  = np.atleast_2d(left_seq)
        right_seq = np.atleast_2d(right_seq)

        rec = self._init_rerun()

        # 设置蓝图
        blueprint = self._make_blueprint(has_left=True, has_right=True)
        rec.send_blueprint(blueprint)

        # 左手
        left_loader = UrdfHandLoader(self.config, side="left",  entity_root="hand/left")
        left_loader.log_static(rec)
        left_loader.log_sequence(rec, left_seq, fps=fps, timeline=timeline)

        # 右手（偏移一段距离，避免重叠）
        right_loader = UrdfHandLoader(self.config, side="right", entity_root="hand/right")
        right_loader.log_static(rec)
        right_loader.log_sequence(rec, right_seq, fps=fps, timeline=timeline)

        if save_path is not None:
            save_path = Path(save_path)
            rr.save(str(save_path), default_blueprint=blueprint)
            print(f"[✓] 已保存录制文件: {save_path}")

    def save_rrd(
        self,
        joint_angles_seq: np.ndarray,
        output_path: Union[str, Path],
        side: str = "left",
        fps: float = 30.0,
    ) -> Path:
        """
        将动画序列保存为 .rrd 文件（不启动 Viewer）。

        Args:
            joint_angles_seq: shape=(T, dof)
            output_path: 输出文件路径
            side: "left" 或 "right"
            fps: 帧率

        Returns:
            保存的文件路径
        """
        self.spawn_viewer = False
        output_path = Path(output_path)
        self.visualize(joint_angles_seq, side=side, fps=fps, save_path=output_path)
        return output_path

    def print_joint_info(self, side: str = "left") -> None:
        """打印关节信息，方便调试"""
        joints = self.config.joints.get(side) or self.config.joints.get("default", [])
        joints = sorted(joints, key=lambda j: j.dof_index)
        print(f"\n{'='*50}")
        print(f"手型: {self.config.display_name}  侧: {side}")
        print(f"DOF: {self.config.dof}  关节数: {len(joints)}")
        print(f"{'='*50}")
        print(f"{'索引':>4}  {'关节名称':<35}  {'手指':<10}  {'下限':>8}  {'上限':>8}")
        print(f"{'-'*80}")
        for j in joints:
            print(
                f"{j.dof_index:>4}  {j.name:<35}  {j.finger:<10}  "
                f"{j.limit_lower:>8.4f}  {j.limit_upper:>8.4f}"
            )
        print(f"{'='*50}\n")
