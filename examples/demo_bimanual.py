"""
双手可视化演示 —— Unitree Dex3-1

左手做抓握动作，右手做波浪动作，Rerun Viewer 分三个面板：
  左手 | 右手 | 双手总览

运行（在项目根目录）：
    python examples/demo_bimanual.py
    python examples/demo_bimanual.py --save   # 保存为 .rrd 不弹窗
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from hand_viz import HandVisualizer

FPS = 30.0
DURATION = 5.0  # 秒
T = int(FPS * DURATION)


def make_left_grasp() -> np.ndarray:
    """左手：张开 → 握紧 → 张开（平滑抓握）"""
    t = np.linspace(0, 1, T)
    grasp = np.sin(np.pi * t) ** 2   # 0→1→0 平滑曲线

    seq = np.zeros((T, 7))
    # 拇指弯曲
    seq[:, 1] = grasp * 0.8
    seq[:, 2] = grasp * 1.5
    # 中指弯曲（左手关节上限为 0，向负方向弯）
    seq[:, 3] = -grasp * 1.4
    seq[:, 4] = -grasp * 1.5
    # 食指弯曲
    seq[:, 5] = -grasp * 1.4
    seq[:, 6] = -grasp * 1.5
    return seq


def make_right_wave() -> np.ndarray:
    """右手：手指依次弯曲（波浪效果）"""
    t = np.linspace(0, 4 * np.pi, T)
    seq = np.zeros((T, 7))

    # 右手关节方向与左手相反（正值弯曲）
    # 中指（关节 3,4）
    seq[:, 3] = 1.2 * np.clip(np.sin(t), 0, 1)
    seq[:, 4] = 1.5 * np.clip(np.sin(t), 0, 1)
    # 食指（关节 5,6），相位延迟 0.8
    seq[:, 5] = 1.2 * np.clip(np.sin(t - 0.8), 0, 1)
    seq[:, 6] = 1.5 * np.clip(np.sin(t - 0.8), 0, 1)
    # 拇指（关节 1,2），相位再延迟
    seq[:, 1] = -0.6 * np.clip(np.sin(t - 1.6), 0, 1)
    seq[:, 2] = -1.2 * np.clip(np.sin(t - 1.6), 0, 1)
    return seq


def main():
    parser = argparse.ArgumentParser(description="双手可视化演示")
    parser.add_argument("--save", action="store_true",
                        help="保存为 .rrd 文件（不弹出 Viewer）")
    args = parser.parse_args()

    left_seq  = make_left_grasp()
    right_seq = make_right_wave()

    print(f"[双手演示] 手型: 宇树 Dex3-1  时长: {DURATION}s @ {FPS}fps")
    print(f"  左手: 抓握动作  shape={left_seq.shape}")
    print(f"  右手: 波浪动作  shape={right_seq.shape}")

    viz = HandVisualizer("unitree_dex3", app_id="bimanual_demo",
                         spawn_viewer=not args.save)

    if args.save:
        out = Path(__file__).parent.parent / "output" / "bimanual_demo.rrd"
        out.parent.mkdir(exist_ok=True)
        viz.visualize_bimanual(left_seq, right_seq, fps=FPS, save_path=out)
        print(f"\n[✓] 已保存: {out}")
        print(f"    打开命令: rerun {out}")
    else:
        viz.visualize_bimanual(left_seq, right_seq, fps=FPS)
        print("\n[✓] Rerun Viewer 已启动")


if __name__ == "__main__":
    main()
