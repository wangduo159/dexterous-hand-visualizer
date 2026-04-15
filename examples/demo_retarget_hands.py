"""
七机型灵巧手 Demo 动画生成脚本
覆盖：inspire / allegro / shadow / svh / leap / ability / panda

每个机型生成三种动作序列（全部保存为 .rrd）：
  grasp  —— 五指同步抓握（张开 → 握紧 → 张开）
  wave   —— 手指依次弯曲（波浪效果）
  pinch  —— 拇指与各指依次对捏

运行方式（在项目根目录执行）：
  # 一键生成所有机型所有动作
  python examples/demo_retarget_hands.py --all

  # 单机型单动作（实时查看）
  python examples/demo_retarget_hands.py --hand shadow_hand --mode wave

  # 单机型保存 rrd
  python examples/demo_retarget_hands.py --hand ability_hand --mode grasp --save

生成的 .rrd 文件保存在 output/demo/ 目录，用以下命令打开：
  rerun output/demo/<hand>_<mode>.rrd
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hand_viz import HandVisualizer
from hand_viz.config_loader import load_hand_config

# ─────────────────────────────────────────────
# 通用工具函数
# ─────────────────────────────────────────────

def smooth_pulse(t_norm: np.ndarray) -> np.ndarray:
    """0→1→0 平滑脉冲（sin²），t_norm ∈ [0,1]"""
    return np.sin(np.pi * t_norm) ** 2


def make_grasp_seq(dof, fps, duration, flex_map, abduct_map=None):
    """
    抓握序列：所有弯曲关节同步 0→max→0。
    flex_map:   {dof_index: amplitude}
    abduct_map: {dof_index: amplitude}（可选，外展/内收）
    """
    T = int(fps * duration)
    t = np.linspace(0, 1, T)
    pulse = smooth_pulse(t)
    seq = np.zeros((T, dof))
    for idx, amp in flex_map.items():
        seq[:, idx] = pulse * amp
    if abduct_map:
        for idx, amp in abduct_map.items():
            seq[:, idx] = pulse * amp
    return seq


def make_wave_seq(dof, fps, duration, finger_groups):
    """
    波浪序列：每根手指依次错开相位弯曲。
    finger_groups: list of (dof_indices, amplitude, phase_offset)
    """
    T = int(fps * duration)
    t = np.linspace(0, 4 * np.pi, T)
    seq = np.zeros((T, dof))
    for dof_indices, amp, phase in finger_groups:
        wave = amp * np.clip(np.sin(t - phase), 0, 1)
        for idx in dof_indices:
            seq[:, idx] = wave
    return seq


def make_pinch_seq(dof, fps, duration, thumb_map, finger_groups):
    """
    对捏序列：拇指保持弯曲，各指依次与拇指对捏。
    thumb_map:     {dof_index: amplitude}  拇指全程保持
    finger_groups: list of (dof_indices, amplitude, phase_offset)  各指依次弯曲
    """
    T = int(fps * duration)
    t = np.linspace(0, 1, T)
    pulse = smooth_pulse(t)
    seq = np.zeros((T, dof))
    # 拇指全程保持弯曲
    for idx, amp in thumb_map.items():
        seq[:, idx] = pulse * amp
    # 各指依次对捏（用 sin² 脉冲，错开相位）
    n = len(finger_groups)
    for i, (dof_indices, amp) in enumerate(finger_groups):
        # 每根手指占 1/n 时间窗口，用高斯脉冲
        center = (i + 0.5) / n
        width = 0.15
        pinch = np.exp(-((t - center) ** 2) / (2 * width ** 2))
        pinch = pinch / pinch.max()
        for idx in dof_indices:
            seq[:, idx] = pinch * amp
    return seq


def _run(viz, seq, side, fps, save, output_dir, name, mode):
    print(f"  shape={seq.shape}  时长={seq.shape[0]/fps:.1f}s  "
          f"范围=[{seq.min():.3f}, {seq.max():.3f}]")
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{name}_{mode}.rrd"
        viz.save_rrd(seq, out, side=side, fps=fps)
        print(f"  [✓] 已保存: {out}")
        print(f"      rerun {out}")
    else:
        viz.visualize(seq, side=side, fps=fps)
        print("  [✓] Rerun Viewer 已启动")


# ─────────────────────────────────────────────
# Inspire Hand  (12 DOF, right)
# 0=thumb_yaw  1=thumb_pitch  2=thumb_inter  3=thumb_distal
# 4=index_prox 5=index_inter
# 6=middle_prox 7=middle_inter
# 8=ring_prox  9=ring_inter
# 10=pinky_prox 11=pinky_inter
# ─────────────────────────────────────────────

def demo_inspire_hand(mode, save, output_dir):
    print("\n" + "="*55)
    print("  Inspire Hand  (12 DOF, right)")
    print("="*55)
    viz = HandVisualizer("inspire_hand", app_id=f"inspire_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 5.0

    if mode == "grasp":
        seq = make_grasp_seq(12, fps, dur,
            flex_map={1:0.5, 2:0.7, 3:0.35, 4:1.3, 5:1.4, 6:1.3, 7:1.4, 8:1.3, 9:1.4, 10:1.3, 11:1.4},
            abduct_map={0: 0.8})

    elif mode == "wave":
        seq = make_wave_seq(12, fps, dur, [
            ([1,2,3],   0.6, 0.0),
            ([4,5],     1.3, 1.05),
            ([6,7],     1.3, 2.09),
            ([8,9],     1.3, 3.14),
            ([10,11],   1.3, 4.19),
        ])

    else:  # pinch
        seq = make_pinch_seq(12, fps, dur,
            thumb_map={0:0.8, 1:0.4, 2:0.6, 3:0.3},
            finger_groups=[
                ([4,5],   1.3),
                ([6,7],   1.3),
                ([8,9],   1.3),
                ([10,11], 1.3),
            ])

    _run(viz, seq, "right", fps, save, output_dir, "inspire_hand", mode)


# ─────────────────────────────────────────────
# Allegro Hand  (16 DOF, right)
# 0~3: index(侧偏+3关节)  4~7: middle  8~11: ring
# 12~15: thumb(特殊方向)
# ─────────────────────────────────────────────

def demo_allegro_hand(mode, save, output_dir):
    print("\n" + "="*55)
    print("  Allegro Hand  (16 DOF, right)")
    print("="*55)
    viz = HandVisualizer("allegro_hand", app_id=f"allegro_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 5.0

    if mode == "grasp":
        seq = make_grasp_seq(16, fps, dur,
            flex_map={
                1:1.4, 2:1.5, 3:1.4,   # index 弯曲
                5:1.4, 6:1.5, 7:1.4,   # middle
                9:1.4, 10:1.5, 11:1.4, # ring
                12:0.8, 13:0.8, 14:1.4, 15:1.5,  # thumb
            },
            abduct_map={0:-0.2, 4:-0.1, 8:0.1})

    elif mode == "wave":
        seq = make_wave_seq(16, fps, dur, [
            ([12,13,14,15], 0.9, 0.0),   # thumb
            ([1,2,3],       1.4, 1.05),  # index
            ([5,6,7],       1.4, 2.09),  # middle
            ([9,10,11],     1.4, 3.14),  # ring
        ])

    else:  # pinch
        seq = make_pinch_seq(16, fps, dur,
            thumb_map={12:0.7, 13:0.7, 14:1.2, 15:1.3},
            finger_groups=[
                ([1,2,3],   1.4),
                ([5,6,7],   1.4),
                ([9,10,11], 1.4),
            ])

    _run(viz, seq, "right", fps, save, output_dir, "allegro_hand", mode)


# ─────────────────────────────────────────────
# Shadow Hand  (24 DOF, right)
# 0=WRJ2  1=WRJ1（手腕，保持0）
# 2=FFJ4(侧) 3=FFJ3 4=FFJ2 5=FFJ1  食指
# 6=LFJ5(掌) 7=LFJ4(侧) 8=LFJ3 9=LFJ2 10=LFJ1  小指
# 11=MFJ4(侧) 12=MFJ3 13=MFJ2 14=MFJ1  中指
# 15=RFJ4(侧) 16=RFJ3 17=RFJ2 18=RFJ1  无名指
# 19=THJ5(旋) 20=THJ4 21=THJ3 22=THJ2 23=THJ1  拇指
# ─────────────────────────────────────────────

def demo_shadow_hand(mode, save, output_dir):
    print("\n" + "="*55)
    print("  Shadow Hand  (24 DOF, right)")
    print("="*55)
    viz = HandVisualizer("shadow_hand", app_id=f"shadow_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 5.0

    if mode == "grasp":
        seq = make_grasp_seq(24, fps, dur,
            flex_map={
                3:1.4, 4:1.4, 5:1.4,        # FF 食指
                8:1.4, 9:1.4, 10:1.4,        # LF 小指
                12:1.4, 13:1.4, 14:1.4,      # MF 中指
                16:1.4, 17:1.4, 18:1.4,      # RF 无名指
                20:1.0, 22:0.5, 23:1.3,      # TH 拇指弯曲
            },
            abduct_map={19:0.6, 6:0.5})      # 拇指旋转 + 小指掌骨

    elif mode == "wave":
        seq = make_wave_seq(24, fps, dur, [
            ([20,22,23],      1.0, 0.0),    # 拇指
            ([3,4,5],         1.4, 1.05),   # 食指
            ([12,13,14],      1.4, 2.09),   # 中指
            ([16,17,18],      1.4, 3.14),   # 无名指
            ([8,9,10],        1.4, 4.19),   # 小指
        ])

    else:  # pinch
        seq = make_pinch_seq(24, fps, dur,
            thumb_map={19:0.6, 20:0.9, 22:0.4, 23:1.2},
            finger_groups=[
                ([3,4,5],    1.4),
                ([12,13,14], 1.4),
                ([16,17,18], 1.4),
                ([8,9,10],   1.4),
            ])

    _run(viz, seq, "right", fps, save, output_dir, "shadow_hand", mode)


# ─────────────────────────────────────────────
# Schunk SVH  (20 DOF, right)
# 0=Thumb_Opposition  1=Thumb_Flexion  2=j3  3=j4  拇指
# 4=index_spread  5=Index_Prox  6=Index_Distal  7=j14  食指
# 8=j5(掌骨)  9=Finger_Spread  中指掌
# 10=Pinky  11=j13  12=j17  小指
# 13=ring_spread  14=Ring_Finger  15=j12  16=j16  无名指
# 17=Middle_Prox  18=Middle_Distal  19=j15  中指
# ─────────────────────────────────────────────

def demo_svh_hand(mode, save, output_dir):
    print("\n" + "="*55)
    print("  Schunk SVH  (20 DOF, right)")
    print("="*55)
    viz = HandVisualizer("svh_hand", app_id=f"svh_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 5.0

    if mode == "grasp":
        seq = make_grasp_seq(20, fps, dur,
            flex_map={
                1:0.8, 2:0.7, 3:1.2,         # 拇指弯曲
                5:0.7, 6:1.1, 7:1.2,          # 食指
                10:0.8, 11:1.1, 12:1.2,       # 小指
                14:0.8, 15:1.1, 16:1.2,       # 无名指
                17:0.7, 18:1.1, 19:1.2,       # 中指
            },
            abduct_map={0:0.7})               # 拇指对掌

    elif mode == "wave":
        seq = make_wave_seq(20, fps, dur, [
            ([1,2,3],      0.9, 0.0),    # 拇指
            ([5,6,7],      0.9, 1.05),   # 食指
            ([17,18,19],   0.9, 2.09),   # 中指
            ([14,15,16],   0.9, 3.14),   # 无名指
            ([10,11,12],   0.9, 4.19),   # 小指
        ])

    else:  # pinch
        seq = make_pinch_seq(20, fps, dur,
            thumb_map={0:0.6, 1:0.7, 2:0.6, 3:1.0},
            finger_groups=[
                ([5,6,7],    0.9),
                ([17,18,19], 0.9),
                ([14,15,16], 0.9),
                ([10,11,12], 0.9),
            ])

    _run(viz, seq, "right", fps, save, output_dir, "svh_hand", mode)


# ─────────────────────────────────────────────
# LEAP Hand  (16 DOF, right)
# 0=侧偏 1=近端 2=中端 3=远端  食指
# 4=侧偏 5=近端 6=中端 7=远端  中指
# 8=侧偏 9=近端 10=中端 11=远端 无名指
# 12=旋转 13=侧偏 14=近端 15=远端  拇指
# ─────────────────────────────────────────────

def demo_leap_hand(mode, save, output_dir):
    print("\n" + "="*55)
    print("  LEAP Hand  (16 DOF, right)")
    print("="*55)
    viz = HandVisualizer("leap_hand", app_id=f"leap_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 5.0

    if mode == "grasp":
        seq = make_grasp_seq(16, fps, dur,
            flex_map={
                1:1.8, 2:1.5, 3:1.6,    # 食指
                5:1.8, 6:1.5, 7:1.6,    # 中指
                9:1.8, 10:1.5, 11:1.6,  # 无名指
                12:1.6, 13:1.8,          # 拇指旋转+侧偏（正值=对掌方向）
                14:1.5, 15:1.5,          # 拇指弯曲
            })

    elif mode == "wave":
        seq = make_wave_seq(16, fps, dur, [
            ([12,13,14,15], 1.4, 0.0),   # 拇指（正值对掌）
            ([1,2,3],       1.6, 1.05),  # 食指
            ([5,6,7],       1.6, 2.09),  # 中指
            ([9,10,11],     1.6, 3.14),  # 无名指
        ])

    else:  # pinch
        seq = make_pinch_seq(16, fps, dur,
            thumb_map={12:1.5, 13:1.8, 14:1.3, 15:1.3},  # 正值=对掌方向
            finger_groups=[
                ([1,2,3],   1.6),
                ([5,6,7],   1.6),
                ([9,10,11], 1.6),
            ])

    _run(viz, seq, "right", fps, save, output_dir, "leap_hand", mode)


# ─────────────────────────────────────────────
# Ability Hand  (10 DOF, right)
# 0=index_q1  1=index_q2
# 2=middle_q1 3=middle_q2
# 4=pinky_q1  5=pinky_q2
# 6=ring_q1   7=ring_q2
# 8=thumb_q1(-) 9=thumb_q2(+)
# ─────────────────────────────────────────────

def demo_ability_hand(mode, save, output_dir):
    print("\n" + "="*55)
    print("  Ability Hand  (10 DOF, right)")
    print("="*55)
    viz = HandVisualizer("ability_hand", app_id=f"ability_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 5.0

    if mode == "grasp":
        seq = make_grasp_seq(10, fps, dur,
            flex_map={
                0:1.8, 1:2.2,   # index
                2:1.8, 3:2.2,   # middle
                4:1.6, 5:2.0,   # pinky
                6:1.8, 7:2.2,   # ring
                9:1.8,           # thumb_q2
            },
            abduct_map={8:-1.8})  # thumb_q1（负值方向）

    elif mode == "wave":
        seq = make_wave_seq(10, fps, dur, [
            ([9],       1.6, 0.0),    # 拇指
            ([0,1],     1.8, 1.05),   # 食指
            ([2,3],     1.8, 2.09),   # 中指
            ([6,7],     1.8, 3.14),   # 无名指
            ([4,5],     1.8, 4.19),   # 小指
        ])
        # thumb_q1 全程保持对掌位置
        T = seq.shape[0]
        t = np.linspace(0, 1, T)
        seq[:, 8] = -smooth_pulse(t) * 1.5

    else:  # pinch
        seq = make_pinch_seq(10, fps, dur,
            thumb_map={8:-1.6, 9:1.5},
            finger_groups=[
                ([0,1], 1.8),
                ([2,3], 1.8),
                ([6,7], 1.8),
                ([4,5], 1.8),
            ])

    _run(viz, seq, "right", fps, save, output_dir, "ability_hand", mode)


# ─────────────────────────────────────────────
# Panda Gripper  (2 DOF, prismatic, 单位=米)
# 0=panda_finger_joint1  [0, 0.04m]
# 1=panda_finger_joint2  [0, 0.04m]
# ─────────────────────────────────────────────

def demo_panda_gripper(mode, save, output_dir):
    print("\n" + "="*55)
    print("  Panda Gripper  (2 DOF prismatic, right)")
    print("="*55)
    viz = HandVisualizer("panda_gripper", app_id=f"panda_{mode}", spawn_viewer=not save)
    fps, dur = 30.0, 4.0
    T = int(fps * dur)
    t = np.linspace(0, 1, T)
    seq = np.zeros((T, 2))

    if mode == "grasp":
        # 两指同步张开→夹紧→张开
        pulse = smooth_pulse(t)
        seq[:, 0] = pulse * 0.038
        seq[:, 1] = pulse * 0.038

    elif mode == "wave":
        # 两指交替开合（模拟"夹爪波浪"）
        t2 = np.linspace(0, 6 * np.pi, T)
        seq[:, 0] = 0.035 * np.clip(np.sin(t2), 0, 1)
        seq[:, 1] = 0.035 * np.clip(np.sin(t2 - np.pi), 0, 1)

    else:  # pinch — 快速多次夹取
        t3 = np.linspace(0, 4 * np.pi, T)
        pulse3 = np.clip(np.sin(t3), 0, 1) ** 2
        seq[:, 0] = pulse3 * 0.036
        seq[:, 1] = pulse3 * 0.036

    _run(viz, seq, "right", fps, save, output_dir, "panda_gripper", mode)


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

DEMOS = {
    "inspire_hand":  demo_inspire_hand,
    "allegro_hand":  demo_allegro_hand,
    "shadow_hand":   demo_shadow_hand,
    "svh_hand":      demo_svh_hand,
    "leap_hand":     demo_leap_hand,
    "ability_hand":  demo_ability_hand,
    "panda_gripper": demo_panda_gripper,
}

MODES = ["grasp", "wave", "pinch"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="七机型灵巧手 Demo 动画生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 一键生成所有机型所有动作（21 个 rrd）
  python examples/demo_retarget_hands.py --all

  # 单机型单动作实时查看
  python examples/demo_retarget_hands.py --hand shadow_hand --mode wave

  # 单机型保存 rrd
  python examples/demo_retarget_hands.py --hand ability_hand --mode grasp --save
        """
    )
    parser.add_argument("--hand", choices=list(DEMOS.keys()), help="指定机型")
    parser.add_argument("--mode", choices=MODES, default="grasp",
                        help="动作模式：grasp/wave/pinch（默认 grasp）")
    parser.add_argument("--save", action="store_true", help="保存为 .rrd 而非实时显示")
    parser.add_argument("--all",  action="store_true", help="生成所有机型所有动作")

    args = parser.parse_args()
    output_dir = Path(__file__).parent.parent / "output" / "demo"

    if args.all:
        print(f"\n[批量] 生成 {len(DEMOS)} 机型 × {len(MODES)} 动作 = {len(DEMOS)*len(MODES)} 个 rrd")
        for hand_name, demo_fn in DEMOS.items():
            for mode in MODES:
                try:
                    demo_fn(mode, save=True, output_dir=output_dir)
                except Exception as e:
                    print(f"  [ERROR] {hand_name}/{mode}: {e}")
                    import traceback; traceback.print_exc()
        print(f"\n[✓] 全部完成，文件保存在: {output_dir}")
        print(f"    打开示例: rerun {output_dir}/shadow_hand_wave.rrd")

    elif args.hand:
        DEMOS[args.hand](args.mode, save=args.save, output_dir=output_dir)

    else:
        parser.print_help()
