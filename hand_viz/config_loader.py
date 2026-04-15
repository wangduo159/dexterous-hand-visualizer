"""
配置文件加载器
负责读取 configs/*.yaml，解析各手型的关节映射信息
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# configs 目录路径
_CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# 支持的手型 ID → 配置文件名
# hand_name 同时也是 urdf_assets/ 下的子目录名（除非 yaml 中有 urdf_dir_override）
HAND_REGISTRY: Dict[str, str] = {
    # 本项目自有手型
    "unitree_dex1_1":  "unitree_dex1_1.yaml",
    "unitree_dex3":    "unitree_dex3.yaml",
    "unitree_dex5_1":  "unitree_dex5_1.yaml",
    "inspire_hand":    "inspire_hand.yaml",
    "brainco_revo2":   "brainco_revo2.yaml",
    "sharpa_wave":     "sharpa_wave.yaml",
    # dex-retargeting 资产手型
    "allegro_hand":    "allegro_hand.yaml",
    "shadow_hand":     "shadow_hand.yaml",
    "svh_hand":        "svh_hand.yaml",
    "leap_hand":       "leap_hand.yaml",
    "ability_hand":    "ability_hand.yaml",
    "panda_gripper":   "panda_gripper.yaml",
}


@dataclass
class JointInfo:
    """单个关节的信息"""
    name: str
    finger: str
    dof_index: int
    limit_lower: float
    limit_upper: float
    description: str = ""


@dataclass
class HandConfig:
    """一种灵巧手的完整配置"""
    hand_name: str
    display_name: str
    dof: int
    fingers: List[str]
    urdf_paths: Dict[str, str]          # side -> urdf 文件名
    joints: Dict[str, List[JointInfo]]  # side -> [JointInfo, ...]
    urdf_dir: Path                      # URDF 文件所在目录
    mesh_format: str = "stl"            # 主要 mesh 格式：stl / glb / obj
    raw: dict = field(default_factory=dict)  # 原始 yaml 内容

    def get_joint_names(self, side: str = "left") -> List[str]:
        """按 dof_index 顺序返回关节名列表"""
        joints = self.joints.get(side) or self.joints.get("default", [])
        return [j.name for j in sorted(joints, key=lambda j: j.dof_index)]

    def get_urdf_path(self, side: str = "left") -> Optional[Path]:
        """返回指定侧 URDF 的绝对路径"""
        filename = self.urdf_paths.get(side) or self.urdf_paths.get("default")
        if filename is None:
            return None
        return self.urdf_dir / filename

    def urdf_exists(self, side: str = "left") -> bool:
        p = self.get_urdf_path(side)
        return p is not None and p.exists()

    def get_mesh_search_dirs(self, side: str = "left") -> List[Path]:
        """
        返回 mesh 文件的搜索目录列表（按优先级排序）。
        不同手型的 mesh 目录结构不同，统一在此处理。
        """
        dirs = []
        # 通用：meshes/ 目录
        dirs.append(self.urdf_dir / "meshes")
        # BrainCo：meshes 按左右手分子目录
        dirs.append(self.urdf_dir / "meshes" / f"revo2_{side}_hand")
        # Inspire Hand：meshes/visual/（GLB）
        dirs.append(self.urdf_dir / "meshes" / "visual")
        # Sharpa Wave：左手 mesh 在 meshes_left/
        if side == "left":
            dirs.append(self.urdf_dir / "meshes_left")
        # 根目录兜底
        dirs.append(self.urdf_dir)
        return [d for d in dirs if d.exists()]


def _parse_joints(joints_raw: dict, side: str) -> List[JointInfo]:
    """解析 yaml 中的 joints.left / joints.right / joints.default"""
    raw_list = joints_raw.get(side) or joints_raw.get("default")
    if not raw_list:
        return []
    result = []
    for item in raw_list:
        result.append(JointInfo(
            name=item["name"],
            finger=item.get("finger", "unknown"),
            dof_index=item.get("dof_index", 0),
            limit_lower=float(item.get("limit_lower", -3.14159)),
            limit_upper=float(item.get("limit_upper",  3.14159)),
            description=item.get("description", ""),
        ))
    return result


def load_hand_config(hand_name: str) -> HandConfig:
    """
    加载指定手型的配置。

    Args:
        hand_name: 手型 ID，如 "unitree_dex3"

    Returns:
        HandConfig 对象

    Raises:
        ValueError: 手型不在注册表中
        FileNotFoundError: 配置文件不存在
    """
    if hand_name not in HAND_REGISTRY:
        available = ", ".join(HAND_REGISTRY.keys())
        raise ValueError(
            f"未知手型: '{hand_name}'。可用手型: {available}"
        )

    config_path = _CONFIGS_DIR / HAND_REGISTRY[hand_name]
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 解析 URDF 路径
    urdf_raw = raw.get("urdf", {})
    urdf_paths = {k: v for k, v in urdf_raw.items() if v is not None}

    # URDF 资产目录：优先使用 yaml 中的 urdf_dir_override，否则用 urdf_assets/<hand_name>
    override = raw.get("urdf_dir_override")
    if override:
        urdf_dir = Path(override)
    else:
        urdf_dir = Path(__file__).parent.parent / "urdf_assets" / hand_name

    # 解析关节
    joints_raw = raw.get("joints", {})
    joints: Dict[str, List[JointInfo]] = {}
    for side in ["left", "right", "default"]:
        parsed = _parse_joints(joints_raw, side)
        if parsed:
            joints[side] = parsed

    # 如果 right 为 None（yaml 中写 ~），自动从 left 生成（替换前缀）
    if "right" not in joints and "left" in joints:
        right_joints = []
        for j in joints["left"]:
            right_name = j.name.replace("left_", "right_", 1)
            right_joints.append(JointInfo(
                name=right_name,
                finger=j.finger,
                dof_index=j.dof_index,
                limit_lower=j.limit_lower,
                limit_upper=j.limit_upper,
                description=j.description,
            ))
        joints["right"] = right_joints

    return HandConfig(
        hand_name=raw.get("hand_name", hand_name),
        display_name=raw.get("display_name", hand_name),
        dof=raw.get("dof", 0),
        fingers=raw.get("fingers", []),
        urdf_paths=urdf_paths,
        joints=joints,
        urdf_dir=urdf_dir,
        mesh_format=raw.get("mesh_format", "stl"),
        raw=raw,
    )


def list_available_hands() -> Dict[str, dict]:
    """
    列出所有已注册的手型及其基本信息。

    Returns:
        dict: {hand_name: {"display_name": ..., "dof": ..., "urdf_ready": bool}}
    """
    result = {}
    for hand_name in HAND_REGISTRY:
        try:
            cfg = load_hand_config(hand_name)
            urdf_ready_left = cfg.urdf_exists("left")
            urdf_ready_right = cfg.urdf_exists("right")
            result[hand_name] = {
                "display_name": cfg.display_name,
                "dof": cfg.dof,
                "urdf_ready_left": urdf_ready_left,
                "urdf_ready_right": urdf_ready_right,
                "mesh_format": cfg.mesh_format,
            }
        except Exception as e:
            result[hand_name] = {"error": str(e)}
    return result
