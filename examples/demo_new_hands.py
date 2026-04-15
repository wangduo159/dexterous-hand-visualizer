"""
新手型可视化演示脚本
覆盖：宇树 Dex1-1 / 宇树 Dex5-1 / Inspire Hand / BrainCo Revo2 / Sharpa Wave

每个手型演示两个动作序列：
  1. grasp  —— 五指同步抓握（张开 → 握紧 → 张开）
  2. wave   —— 手指依次弯曲（波浪效果）

运行方式（在项目根目录执行）：
  # 查看所有可用演示
  python examples/demo_new_hands.py --help

  # 运行单个手型（实时打开 Rerun Viewer）
  python examples/demo_new_hands.py --hand brainco_revo2 --mode grasp

  # 保存为 .rrd 文件（不打开 Viewer）
  python examples/demo_new_hands.py --hand inspire_hand --mode grasp --save

  # 一键跑所有手型，全部保存为 .rrd
  python examples/demo_new_hands.py --all --save

生成的 .rrd 文件保存在项目根目录，用以下命令打开：
  rerun output/<hand_name>_demo.rrd
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hand_viz import HandVisualizer
from hand_viz.config_loader import load_hand_config

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def smooth_pulse(t_norm: np.ndarray) -> np.ndarray:
    """0→1→0 的平滑脉冲曲线（sin²）"""
    return np.sin(np.pi * t_norm) ** 2


def make_grasp_seq(dof: int, fps: float, duration: float,
                   flex_dofs: list, flex_scales: list,
                   abduct_dofs: list = None, abduct_scales: list = None) -> np.ndarray:
    """
    生成抓握序列。

    Args:
        dof:           总 DOF 数
        fps:           帧率
        duration:      时长（秒）
        flex_dofs:     参与弯曲的 dof 索引列表
        flex_scales:   对应的弯曲幅度（弧度）
        abduct_dofs:   外展/内收 dof 索引（可选）
        abduct_scales: 对应幅度
    Returns:
        shape=(T, dof) 的角度序列
    """
    T = int(fps * duration)
    t = np.linspace(0, 1, T)
    grasp = smooth_pulse(t)

    seq = np.zeros((T, dof))
    for idx, scale in zip(flex_dofs, flex_scales):
        seq[:, idx] = grasp * scale
    if abduct_dofs:
        for idx, scale in zip(abduct_dofs, abduct_scales):
            seq[:, idx] = grasp * scale
    return seq


def make_wave_seq(dof: int, fps: float, duration: float,
                  finger_groups: list) -> np.ndarray:
    """
    生成波浪序列。

    Args:
        finger_groups: list of (dof_indices, scale, phase_offset)
                       每组代表一根手指的关节，依次错开相位
    Returns:
        shape=(T, dof) 的角度序列
    """
    T = int(fps * duration)
    t = np.linspace(0, 4 * np.pi, T)
    seq = np.zeros((T, dof))
    for dof_indices, scale, phase in finger_groups:
        wave = scale * np.clip(np.sin(t - phase), 0, 1)
        for idx in dof_indices:
            seq[:, idx] = wave
    return seq


# ─────────────────────────────────────────────
# 各手型演示定义
# ─────────────────────────────────────────────

def demo_unitree_dex5_1(mode: str, save: bool, output_dir: Path):
    """
    宇树 Dex5-1 演示（20 DOF，右手）

    关节顺序（dof_index）：
      拇指:  0=Yaw_11R(外展)  1=Roll_12R(侧偏)  2=Pitch_13R  3=Pitch_14R
      食指:  4=Roll_21R(侧偏) 5=Pitch_22R       6=Pitch_23R  7=Pitch_24R
      中指:  8=Roll_31R       9=Pitch_32R       10=Pitch_33R 11=Pitch_34R
      无名:  12=Roll_41R      13=Pitch_42R      14=Pitch_43R 15=Pitch_44R
      小指:  16=Roll_51R      17=Pitch_52R      18=Pitch_53R 19=Pitch_54R
    """
    print("\n" + "="*55)
    print("  宇树 Dex5-1  (20 DOF, 右手)")
    print("="*55)

    viz = HandVisualizer("unitree_dex5_1", app_id="dex5_demo",
                         spawn_viewer=not save)
    viz.print_joint_info(side="right")

    fps, duration = 30.0, 5.0

    if mode == "grasp":
        print("[动作] 五指同步抓握")
        # 弯曲关节：每指的 Pitch 关节（跳过 Roll/Yaw 外展关节）
        flex_dofs   = [2, 3,   5, 6, 7,   9, 10, 11,   13, 14, 15,   17, 18, 19]
        flex_scales = [1.5, 1.4,  1.4, 1.5, 1.2,  1.4, 1.5, 1.2,  1.4, 1.5, 1.2,  1.4, 1.5, 1.2]
        # 拇指外展配合
        abduct_dofs   = [0, 1]
        abduct_scales = [0.4, 0.8]
        seq = make_grasp_seq(20, fps, duration, flex_dofs, flex_scales,
                             abduct_dofs, abduct_scales)

    else:  # wave
        print("[动作] 手指波浪")
        # 每根手指的弯曲关节组，依次错开 π/3 相位
        finger_groups = [
            ([2, 3],         1.4, 0.0),   # 拇指
            ([5, 6, 7],      1.3, 1.05),  # 食指
            ([9, 10, 11],    1.3, 2.09),  # 中指
            ([13, 14, 15],   1.3, 3.14),  # 无名指
            ([17, 18, 19],   1.3, 4.19),  # 小指
        ]
        seq = make_wave_seq(20, fps, duration, finger_groups)

    _run(viz, seq, side="right", fps=fps, save=save,
         output_dir=output_dir, name="unitree_dex5_1", mode=mode)


def demo_inspire_hand(mode: str, save: bool, output_dir: Path):
    """
    Inspire Hand 演示（12 DOF，右手）

    关节顺序（dof_index）：
      拇指:  0=thumb_proximal_yaw  1=thumb_proximal_pitch
             2=thumb_intermediate  3=thumb_distal
      食指:  4=index_proximal      5=index_intermediate
      中指:  6=middle_proximal     7=middle_intermediate
      无名:  8=ring_proximal       9=ring_intermediate
      小指:  10=pinky_proximal     11=pinky_intermediate
    """
    print("\n" + "="*55)
    print("  Inspire Hand  (12 DOF, 右手)")
    print("="*55)

    viz = HandVisualizer("inspire_hand", app_id="inspire_demo",
                         spawn_viewer=not save)
    viz.print_joint_info(side="right")

    fps, duration = 30.0, 5.0

    if mode == "grasp":
        print("[动作] 五指同步抓握")
        flex_dofs   = [0, 1, 2, 3,   4, 5,   6, 7,   8, 9,   10, 11]
        flex_scales = [0.9, 0.5, 0.7, 0.35,  1.3, 1.4,  1.3, 1.4,  1.3, 1.4,  1.3, 1.4]
        seq = make_grasp_seq(12, fps, duration, flex_dofs, flex_scales)

    else:  # wave
        print("[动作] 手指波浪")
        finger_groups = [
            ([1, 2, 3],   0.6, 0.0),    # 拇指弯曲
            ([4, 5],      1.3, 1.05),   # 食指
            ([6, 7],      1.3, 2.09),   # 中指
            ([8, 9],      1.3, 3.14),   # 无名指
            ([10, 11],    1.3, 4.19),   # 小指
        ]
        seq = make_wave_seq(12, fps, duration, finger_groups)

    _run(viz, seq, side="right", fps=fps, save=save,
         output_dir=output_dir, name="inspire_hand", mode=mode)


def demo_brainco_revo2(mode: str, save: bool, output_dir: Path):
    """
    BrainCo Revo2 演示（11 DOF，右手）

    关节顺序（dof_index）：
      拇指:  0=right_thumb_metacarpal  1=right_thumb_proximal  2=right_thumb_distal
      食指:  3=right_index_proximal    4=right_index_distal
      中指:  5=right_middle_proximal   6=right_middle_distal
      无名:  7=right_ring_proximal     8=right_ring_distal
      小指:  9=right_pinky_proximal    10=right_pinky_distal
    """
    print("\n" + "="*55)
    print("  BrainCo Revo2  (11 DOF, 右手)")
    print("="*55)

    viz = HandVisualizer("brainco_revo2", app_id="brainco_demo",
                         spawn_viewer=not save)
    viz.print_joint_info(side="right")

    fps, duration = 30.0, 5.0

    if mode == "grasp":
        print("[动作] 五指同步抓握")
        flex_dofs   = [0, 1, 2,   3, 4,   5, 6,   7, 8,   9, 10]
        flex_scales = [1.2, 0.9, 0.9,  1.2, 1.4,  1.2, 1.4,  1.2, 1.4,  1.2, 1.4]
        seq = make_grasp_seq(11, fps, duration, flex_dofs, flex_scales)

    else:  # wave
        print("[动作] 手指波浪")
        finger_groups = [
            ([1, 2],    0.9, 0.0),    # 拇指
            ([3, 4],    1.3, 1.05),   # 食指
            ([5, 6],    1.3, 2.09),   # 中指
            ([7, 8],    1.3, 3.14),   # 无名指
            ([9, 10],   1.3, 4.19),   # 小指
        ]
        seq = make_wave_seq(11, fps, duration, finger_groups)

    _run(viz, seq, side="right", fps=fps, save=save,
         output_dir=output_dir, name="brainco_revo2", mode=mode)


def demo_sharpa_wave(mode: str, save: bool, output_dir: Path):
    """
    Sharpa Wave 演示（22 DOF，右手）

    关节顺序（dof_index）：
      拇指:  0=CMC_FE  1=CMC_AA  2=MCP_FE  3=MCP_AA  4=IP
      食指:  5=MCP_FE  6=MCP_AA  7=PIP     8=DIP
      中指:  9=MCP_FE  10=MCP_AA 11=PIP    12=DIP
      无名:  13=MCP_FE 14=MCP_AA 15=PIP    16=DIP
      小指:  17=CMC    18=MCP_FE 19=MCP_AA 20=PIP    21=DIP
    """
    print("\n" + "="*55)
    print("  Sharpa Wave  (22 DOF, 右手)")
    print("="*55)

    viz = HandVisualizer("sharpa_wave", app_id="sharpa_demo",
                         spawn_viewer=not save)
    viz.print_joint_info(side="right")

    fps, duration = 30.0, 5.0

    if mode == "grasp":
        print("[动作] 五指同步抓握（含 AA 外展配合）")
        # 弯曲关节
        flex_dofs   = [0, 2, 4,   5, 7, 8,   9, 11, 12,   13, 15, 16,   17, 18, 20, 21]
        flex_scales = [1.2, 1.0, 1.5,  1.2, 1.5, 1.2,  1.2, 1.5, 1.2,  1.2, 1.5, 1.2,  0.2, 1.2, 1.5, 1.2]
        # AA 外展（抓握时手指略微内收）
        abduct_dofs   = [1, 3,   6,   10,   14,   19]
        abduct_scales = [-0.2, -0.2,  -0.15,  -0.1,  0.1,  0.15]
        seq = make_grasp_seq(22, fps, duration, flex_dofs, flex_scales,
                             abduct_dofs, abduct_scales)

    else:  # wave
        print("[动作] 手指波浪（含 MCP 外展摆动）")
        finger_groups = [
            ([2, 4],      1.2, 0.0),    # 拇指
            ([5, 7, 8],   1.3, 1.05),   # 食指
            ([9, 11, 12], 1.3, 2.09),   # 中指
            ([13,15, 16], 1.3, 3.14),   # 无名指
            ([18,20, 21], 1.3, 4.19),   # 小指
        ]
        seq = make_wave_seq(22, fps, duration, finger_groups)

    _run(viz, seq, side="right", fps=fps, save=save,
         output_dir=output_dir, name="sharpa_wave", mode=mode)


# ─────────────────────────────────────────────
# 运行 & 保存
# ─────────────────────────────────────────────

def _run(viz: HandVisualizer, seq: np.ndarray, side: str,
         fps: float, save: bool, output_dir: Path, name: str, mode: str):
    print(f"[数据] shape={seq.shape}  时长={seq.shape[0]/fps:.1f}s @ {fps}fps")
    print(f"[数据] 角度范围: min={seq.min():.3f}  max={seq.max():.3f} rad")

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{name}_{mode}.rrd"
        viz.save_rrd(seq, out, side=side, fps=fps)
        print(f"\n[✓] 已保存: {out}")
        print(f"    打开命令: rerun {out}")
    else:
        viz.visualize(seq, side=side, fps=fps)
        print("\n[✓] Rerun Viewer 已启动，请在弹出窗口中查看动画")


def demo_unitree_dex1_1(mode: str, save: bool, output_dir: Path):
    """
    宇树 Dex1-1 演示（2 DOF，双指夹爪）

    关节类型： prismatic（线性平移），单位是米
      0 = Joint1_1  手指 1 平移（-0.02m ~ +0.0245m）
      1 = Joint2_1  手指 2 平移（-0.02m ~ +0.0245m）
    """
    print("\n" + "="*55)
    print("  宇树 Dex1-1  (2 DOF, 双指夹爪)")
    print("="*55)

    viz = HandVisualizer("unitree_dex1_1", app_id="dex1_demo",
                         spawn_viewer=not save)
    viz.print_joint_info(side="left")

    fps, duration = 30.0, 4.0
    T = int(fps * duration)
    t = np.linspace(0, 1, T)
    pulse = smooth_pulse(t)

    seq = np.zeros((T, 2))
    if mode == "grasp":
        print("[动作] 夹爪张开 → 夹紧 → 张开")
        # 两指同时向内夹紧（负值 = 内收）
        seq[:, 0] = -pulse * 0.018
        seq[:, 1] = -pulse * 0.018
    else:  # wave
        print("[动作] 两指交替开合")
        t2 = np.linspace(0, 4 * np.pi, T)
        seq[:, 0] = -0.015 * np.clip(np.sin(t2), 0, 1)
        seq[:, 1] = -0.015 * np.clip(np.sin(t2 - np.pi), 0, 1)

    _run(viz, seq, side="left", fps=fps, save=save,
         output_dir=output_dir, name="unitree_dex1_1", mode=mode)


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

DEMOS = {
    "unitree_dex1_1": demo_unitree_dex1_1,
    "unitree_dex5_1": demo_unitree_dex5_1,
    "inspire_hand":   demo_inspire_hand,
    "brainco_revo2":  demo_brainco_revo2,
    "sharpa_wave":    demo_sharpa_wave,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="新手型可视化演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 实时查看 BrainCo 抓握动作
  python examples/demo_new_hands.py --hand brainco_revo2 --mode grasp

  # 保存 Sharpa Wave 波浪动作为 .rrd
  python examples/demo_new_hands.py --hand sharpa_wave --mode wave --save

  # 一键保存所有手型的抓握演示
  python examples/demo_new_hands.py --all --mode grasp --save
        """
    )
    parser.add_argument(
        "--hand",
        choices=list(DEMOS.keys()),
        help="指定手型"
    )
    parser.add_argument(
        "--mode",
        choices=["grasp", "wave"],
        default="grasp",
        help="动作模式：grasp=抓握, wave=波浪（默认: grasp）"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存为 .rrd 文件而非实时显示"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有手型演示"
    )

    args = parser.parse_args()
    output_dir = Path(__file__).parent.parent / "output"

    if args.all:
        print(f"\n[批量模式] 运行所有手型，mode={args.mode}, save={args.save}")
        for hand_name, demo_fn in DEMOS.items():
            try:
                demo_fn(args.mode, save=True, output_dir=output_dir)
            except Exception as e:
                print(f"[ERROR] {hand_name}: {e}")
                import traceback; traceback.print_exc()
    elif args.hand:
        DEMOS[args.hand](args.mode, save=args.save, output_dir=output_dir)
    else:
        parser.print_help()
        print("\n[提示] 未指定 --hand，默认运行 brainco_revo2 grasp 演示")
        demo_brainco_revo2("grasp", save=False, output_dir=output_dir)
