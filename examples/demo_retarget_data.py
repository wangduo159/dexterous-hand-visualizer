"""
示例：对接 dex-retargeting 输出数据并可视化

这个脚本展示如何将 dex-retargeting 库的输出结果
直接送入 HandVisualizer 进行可视化。

dex-retargeting 输出格式:
    retargeting.retarget(positions)  →  np.ndarray, shape=(dof,)
    关节顺序由 retargeting.joint_names 决定

运行:
    python examples/demo_retarget_data.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hand_viz import HandVisualizer, load_hand_config


def align_retarget_to_config(
    retarget_joint_names: list,
    retarget_angles: np.ndarray,
    hand_name: str,
    side: str = "left",
) -> np.ndarray:
    """
    将 dex-retargeting 输出的关节角度，按照 config 中的 dof_index 顺序重排。

    Args:
        retarget_joint_names: retargeting 输出的关节名列表
        retarget_angles: shape=(T, len(retarget_joint_names)) 或 (len(...),)
        hand_name: 手型 ID
        side: "left" 或 "right"

    Returns:
        重排后的角度数组，shape=(T, config.dof) 或 (config.dof,)
    """
    config = load_hand_config(hand_name)
    config_joint_names = config.get_joint_names(side)

    single_frame = retarget_angles.ndim == 1
    if single_frame:
        retarget_angles = retarget_angles[np.newaxis, :]

    T, _ = retarget_angles.shape
    dof = len(config_joint_names)
    aligned = np.zeros((T, dof))

    # 建立 retarget 关节名 → 列索引 的映射
    retarget_name_to_idx = {name: i for i, name in enumerate(retarget_joint_names)}

    matched = 0
    for config_idx, joint_name in enumerate(config_joint_names):
        if joint_name in retarget_name_to_idx:
            retarget_idx = retarget_name_to_idx[joint_name]
            aligned[:, config_idx] = retarget_angles[:, retarget_idx]
            matched += 1
        else:
            print(f"[警告] 关节 '{joint_name}' 在 retarget 输出中未找到，使用 0")

    print(f"[对齐] 成功匹配 {matched}/{dof} 个关节")

    return aligned[0] if single_frame else aligned


def demo_with_mock_retarget():
    """
    模拟 dex-retargeting 输出并可视化（无需真实 retarget 库）
    """
    hand_name = "unitree_dex3"
    side = "left"

    config = load_hand_config(hand_name)
    config_joint_names = config.get_joint_names(side)

    # 模拟 retarget 输出（假设顺序与 config 相同）
    fps = 30.0
    T = int(fps * 4)
    t = np.linspace(0, 2 * np.pi, T)

    # 模拟 retarget 输出的关节名（可能顺序不同）
    mock_retarget_names = [
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ]

    # 模拟 retarget 输出的角度（顺序与 mock_retarget_names 对应）
    mock_angles = np.zeros((T, len(mock_retarget_names)))
    mock_angles[:, 0] = -0.8 * np.abs(np.sin(t))        # index_0
    mock_angles[:, 1] = -1.2 * np.abs(np.sin(t))        # index_1
    mock_angles[:, 2] = -0.8 * np.abs(np.sin(t + 0.3)) # middle_0
    mock_angles[:, 3] = -1.2 * np.abs(np.sin(t + 0.3)) # middle_1
    mock_angles[:, 4] = 0.2 * np.sin(t)                 # thumb_0
    mock_angles[:, 5] = 0.6 * np.abs(np.sin(t))         # thumb_1
    mock_angles[:, 6] = 1.0 * np.abs(np.sin(t))         # thumb_2

    # 对齐到 config 顺序
    aligned_angles = align_retarget_to_config(
        mock_retarget_names, mock_angles, hand_name, side
    )

    print(f"[对齐后] shape={aligned_angles.shape}")
    print(f"[关节顺序] {config_joint_names}")

    # 可视化
    viz = HandVisualizer(hand_name, app_id="retarget_demo")
    viz.visualize(aligned_angles, side=side, fps=fps)


def demo_load_from_file():
    """
    从文件加载 retarget 结果并可视化

    假设文件格式为 .npy，shape=(T, dof)，关节顺序已与 config 对齐
    """
    # 这里用随机数据模拟
    hand_name = "unitree_dex3"
    config = load_hand_config(hand_name)

    T = 150
    dof = config.dof
    angles = np.random.uniform(-0.3, 0.3, (T, dof))

    # 实际使用时替换为:
    # angles = np.load("your_retarget_output.npy")

    viz = HandVisualizer(hand_name, app_id="file_demo")
    viz.visualize(angles, side="left", fps=30.0)


if __name__ == "__main__":
    print("=== 演示：对接 dex-retargeting 输出 ===\n")
    demo_with_mock_retarget()
