"""
示例：宇树 Dex3-1 灵巧手可视化演示

运行前请确保:
1. pip install rerun-sdk numpy pyyaml yourdfpy
2. 下载 Dex3-1 URDF 到 urdf_assets/unitree_dex3/（见 README.md）

运行:
    python examples/demo_dex3.py
"""

import sys
from pathlib import Path

import numpy as np

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hand_viz import HandVisualizer


def demo_grasp_sequence():
    """演示一个抓握动作序列"""
    viz = HandVisualizer("unitree_dex3", app_id="dex3_grasp_demo")

    # 打印关节信息
    viz.print_joint_info(side="left")

    fps = 30.0
    T = int(fps * 4)  # 4 秒

    # Dex3-1 左手 7 个关节（按 dof_index 顺序）:
    # 0: left_hand_thumb_0_joint  (拇指外展,  -60° ~ 60°)
    # 1: left_hand_thumb_1_joint  (拇指近端,  -35° ~ 60°)
    # 2: left_hand_thumb_2_joint  (拇指远端,    0° ~ 100°)
    # 3: left_hand_middle_0_joint (中指近端,  -90° ~ 0°)
    # 4: left_hand_middle_1_joint (中指远端, -100° ~ 0°)
    # 5: left_hand_index_0_joint  (食指近端,  -90° ~ 0°)
    # 6: left_hand_index_1_joint  (食指远端, -100° ~ 0°)

    t = np.linspace(0, 1, T)

    # 平滑的抓握动作：从张开到握紧再到张开
    def smooth_grasp(t_norm):
        """0→1→0 的平滑曲线"""
        return np.sin(np.pi * t_norm) ** 2

    grasp = smooth_grasp(t)

    seq = np.zeros((T, 7))
    # 拇指外展（保持中立）
    seq[:, 0] = 0.0
    # 拇指弯曲
    seq[:, 1] = grasp * 0.8
    seq[:, 2] = grasp * 1.5
    # 中指弯曲（注意：左手中指关节上限为 0，下限为负值）
    seq[:, 3] = -grasp * 1.4
    seq[:, 4] = -grasp * 1.5
    # 食指弯曲
    seq[:, 5] = -grasp * 1.4
    seq[:, 6] = -grasp * 1.5

    print(f"[演示] 抓握序列：{T} 帧，{T/fps:.1f}s")
    viz.visualize(seq, side="left", fps=fps)


def demo_finger_wave():
    """演示手指波浪动作"""
    viz = HandVisualizer("unitree_dex3", app_id="dex3_wave_demo")

    fps = 30.0
    T = int(fps * 6)
    t = np.linspace(0, 4 * np.pi, T)

    seq = np.zeros((T, 7))
    # 每根手指依次弯曲（波浪效果）
    # 中指（关节 3,4）
    seq[:, 3] = -0.8 * np.clip(np.sin(t), 0, 1)
    seq[:, 4] = -1.0 * np.clip(np.sin(t), 0, 1)
    # 食指（关节 5,6），相位延迟
    seq[:, 5] = -0.8 * np.clip(np.sin(t - 0.8), 0, 1)
    seq[:, 6] = -1.0 * np.clip(np.sin(t - 0.8), 0, 1)
    # 拇指（关节 1,2），相位再延迟
    seq[:, 1] = 0.6 * np.clip(np.sin(t - 1.6), 0, 1)
    seq[:, 2] = 1.2 * np.clip(np.sin(t - 1.6), 0, 1)

    print(f"[演示] 波浪序列：{T} 帧，{T/fps:.1f}s")
    viz.visualize(seq, side="left", fps=fps)


def demo_save_rrd():
    """将演示保存为 .rrd 文件"""
    viz = HandVisualizer("unitree_dex3", app_id="dex3_saved")

    fps = 30.0
    T = int(fps * 3)
    t = np.linspace(0, 2 * np.pi, T)

    seq = np.zeros((T, 7))
    seq[:, 1] = 0.5 * np.sin(t)
    seq[:, 2] = 0.8 * np.sin(t)
    seq[:, 3] = -0.6 * np.abs(np.sin(t))
    seq[:, 5] = -0.6 * np.abs(np.sin(t + 0.5))

    output_path = Path(__file__).parent.parent / "dex3_demo.rrd"
    saved = viz.save_rrd(seq, output_path, side="left", fps=fps)
    print(f"[✓] 已保存: {saved}")
    print(f"    用 Rerun Viewer 打开: rerun {saved}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dex3-1 可视化演示")
    parser.add_argument(
        "--mode",
        choices=["grasp", "wave", "save"],
        default="grasp",
        help="演示模式（默认: grasp）",
    )
    args = parser.parse_args()

    if args.mode == "grasp":
        demo_grasp_sequence()
    elif args.mode == "wave":
        demo_finger_wave()
    elif args.mode == "save":
        demo_save_rrd()
