"""
dex-retargeting pkl 数据加载与关节对齐模块

pkl 文件结构：
    d["data"]                        # List[np.ndarray(DOF,)]，每帧关节角度（弧度）
    d["meta_data"]["joint_names"]    # 关节名列表，前 6 个是 dummy 自由关节（手腕位姿用）
    d["meta_data"]["dof"]            # 总 DOF 数（含 dummy）
    d["wrist_transforms"]            # List[np.ndarray(4,4)]，每帧手腕位姿

使用方式：
    from hand_viz.pkl_loader import load_pkl, align_to_config

    frames, wrists, retarget_names = load_pkl("inspire/right/0.pkl")
    aligned = align_to_config(frames, retarget_names, "inspire_hand", side="right")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# dummy 关节前缀，用于识别并跳过
_DUMMY_PREFIXES = ("dummy_",)
_DUMMY_COUNT = 6  # 固定前 6 个为 dummy


def load_pkl(
    pkl_path: str | Path,
) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    """
    加载 dex-retargeting 输出的 pkl 文件。

    Returns:
        finger_angles: shape=(T, finger_dof)，已去掉前 6 个 dummy 维度
        wrist_transforms: List[np.ndarray(4,4)]，长度=T
        finger_joint_names: 去掉 dummy 后的关节名列表
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl 文件不存在: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    raw_frames: List[np.ndarray] = d["data"]           # List[(DOF,)]
    joint_names: List[str] = d["meta_data"]["joint_names"]
    wrist_transforms: List[np.ndarray] = d.get("wrist_transforms", [])

    # 找到 dummy 关节的数量（通常固定为 6，但做一下自动检测）
    n_dummy = 0
    for name in joint_names:
        if any(name.startswith(p) for p in _DUMMY_PREFIXES):
            n_dummy += 1
        else:
            break
    if n_dummy == 0:
        n_dummy = _DUMMY_COUNT  # 兜底

    finger_names = joint_names[n_dummy:]
    finger_dof = len(finger_names)

    # 堆叠为 (T, DOF) 并切掉 dummy 维度
    all_frames = np.stack(raw_frames, axis=0)           # (T, total_dof)
    finger_angles = all_frames[:, n_dummy:]             # (T, finger_dof)

    T = finger_angles.shape[0]
    print(f"[pkl] 加载 {pkl_path.name}: {T} 帧, {finger_dof} 个手指关节")
    print(f"[pkl] 关节名: {finger_names}")

    return finger_angles, wrist_transforms, finger_names


def align_to_config(
    finger_angles: np.ndarray,
    retarget_joint_names: List[str],
    hand_name: str,
    side: str = "right",
) -> np.ndarray:
    """
    将 pkl 中的关节角度按照 config 中的 dof_index 顺序重排。

    pkl 里的关节名可能带 rh_/lh_ 前缀（shadow），或无前缀（inspire/ability），
    config 里的关节名也可能带 right_/left_ 前缀。
    本函数做宽松匹配：去掉常见前缀后比较核心名称。

    Args:
        finger_angles:        shape=(T, len(retarget_joint_names))
        retarget_joint_names: pkl 中去掉 dummy 后的关节名列表
        hand_name:            config 中的手型 ID
        side:                 "left" 或 "right"

    Returns:
        aligned: shape=(T, config.dof)，未匹配的维度填 0
    """
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from hand_viz.config_loader import load_hand_config

    config = load_hand_config(hand_name)
    config_joint_names = config.get_joint_names(side)
    config_dof = len(config_joint_names)
    T = finger_angles.shape[0]

    aligned = np.zeros((T, config_dof))

    # 构建 retarget 关节名 → 列索引 的映射（原始名 + 去前缀名）
    retarget_map: dict = {}
    for i, name in enumerate(retarget_joint_names):
        retarget_map[name] = i
        # 去掉常见前缀：rh_ lh_ right_ left_ right_hand_ left_hand_
        for prefix in ("rh_", "lh_", "right_hand_", "left_hand_", "right_", "left_"):
            if name.startswith(prefix):
                retarget_map[name[len(prefix):]] = i
                break

    matched, unmatched = 0, []
    for cfg_idx, cfg_name in enumerate(config_joint_names):
        # 尝试直接匹配，再尝试去前缀匹配
        candidates = [cfg_name]
        for prefix in ("rh_", "lh_", "right_hand_", "left_hand_", "right_", "left_"):
            if cfg_name.startswith(prefix):
                candidates.append(cfg_name[len(prefix):])
                break

        found = False
        for c in candidates:
            if c in retarget_map:
                aligned[:, cfg_idx] = finger_angles[:, retarget_map[c]]
                matched += 1
                found = True
                break
        if not found:
            unmatched.append(cfg_name)

    print(f"[align] {hand_name}/{side}: 匹配 {matched}/{config_dof} 个关节")
    if unmatched:
        print(f"[align] 未匹配（填0）: {unmatched}")

    return aligned
