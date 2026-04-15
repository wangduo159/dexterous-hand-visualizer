"""
通用 retarget pkl 可视化脚本

支持 dex-retargeting 输出的所有机型：
  inspire / allegro / shadow / svh / leap / ability / panda

pkl 数据目录结构：
  <data_root>/<机型>/<left|right>/<序号>.pkl

机型 → hand_name 映射：
  inspire  → inspire_hand
  allegro  → allegro_hand
  shadow   → shadow_hand
  svh      → svh_hand
  leap     → leap_hand
  ability  → ability_hand
  panda    → panda_gripper

运行示例（在项目根目录执行）：

  # 可视化单个 pkl（实时打开 Rerun Viewer）
  python examples/viz_retarget_pkl.py \\
      --pkl /Users/wangduo/Downloads/retarget/example/vector_retargeting/output/inspire/right/0.pkl \\
      --hand inspire_hand --side right

  # 保存为 .rrd 文件
  python examples/viz_retarget_pkl.py \\
      --pkl .../inspire/right/0.pkl \\
      --hand inspire_hand --side right --save

  # 批量可视化某机型所有 pkl（保存为 rrd）
  python examples/viz_retarget_pkl.py \\
      --batch .../inspire/right/ \\
      --hand inspire_hand --side right --save

  # 一键跑所有机型的第 0 个 pkl
  python examples/viz_retarget_pkl.py --all --data-root .../output/ --save
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hand_viz import HandVisualizer
from hand_viz.pkl_loader import load_pkl, align_to_config

# pkl 目录中的机型名 → 项目 hand_name 映射
HAND_MAP = {
    "inspire": "inspire_hand",
    "allegro": "allegro_hand",
    "shadow":  "shadow_hand",
    "svh":     "svh_hand",
    "leap":    "leap_hand",
    "ability": "ability_hand",
    "panda":   "panda_gripper",
}


def viz_single_pkl(
    pkl_path: Path,
    hand_name: str,
    side: str,
    save: bool,
    output_dir: Path,
    fps: float = 30.0,
) -> None:
    """可视化单个 pkl 文件"""
    print(f"\n{'='*60}")
    print(f"  {hand_name} / {side} / {pkl_path.name}")
    print(f"{'='*60}")

    # 1. 加载 pkl
    finger_angles, wrist_transforms, retarget_names = load_pkl(pkl_path)
    T = finger_angles.shape[0]
    print(f"[数据] {T} 帧, {len(retarget_names)} 个手指关节")

    # 2. 对齐到 config 关节顺序
    aligned = align_to_config(finger_angles, retarget_names, hand_name, side=side)
    print(f"[对齐] aligned shape={aligned.shape}, "
          f"范围=[{aligned.min():.3f}, {aligned.max():.3f}] rad")

    # 3. 可视化
    viz = HandVisualizer(hand_name, app_id=f"retarget_{hand_name}",
                         spawn_viewer=not save)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = pkl_path.stem  # 序号，如 "0"
        out = output_dir / f"{hand_name}_{side}_{stem}.rrd"
        viz.save_rrd(aligned, out, side=side, fps=fps)
        print(f"[✓] 已保存: {out}")
        print(f"    打开: rerun {out}")
    else:
        viz.visualize(aligned, side=side, fps=fps)
        print("[✓] Rerun Viewer 已启动")


def viz_batch(
    batch_dir: Path,
    hand_name: str,
    side: str,
    save: bool,
    output_dir: Path,
    fps: float = 30.0,
    max_files: int = None,
) -> None:
    """批量可视化目录下所有 pkl"""
    pkl_files = sorted(batch_dir.glob("*.pkl"),
                       key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if max_files:
        pkl_files = pkl_files[:max_files]

    print(f"\n[批量] {batch_dir}  共 {len(pkl_files)} 个文件")
    for pkl_path in pkl_files:
        try:
            viz_single_pkl(pkl_path, hand_name, side, save=True,
                           output_dir=output_dir, fps=fps)
        except Exception as e:
            print(f"[ERROR] {pkl_path.name}: {e}")
            import traceback; traceback.print_exc()


def viz_all_hands(
    data_root: Path,
    save: bool,
    output_dir: Path,
    fps: float = 30.0,
    pkl_index: int = 0,
) -> None:
    """对每个机型取第 pkl_index 个 pkl 跑一遍"""
    for folder_name, hand_name in HAND_MAP.items():
        # 找 right 优先，没有就用 left
        for side in ("right", "left"):
            pkl_dir = data_root / folder_name / side
            if not pkl_dir.exists():
                continue
            pkl_files = sorted(pkl_dir.glob("*.pkl"),
                               key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            if not pkl_files:
                continue
            idx = min(pkl_index, len(pkl_files) - 1)
            try:
                viz_single_pkl(pkl_files[idx], hand_name, side,
                               save=True, output_dir=output_dir, fps=fps)
            except Exception as e:
                print(f"[ERROR] {hand_name}/{side}: {e}")
                import traceback; traceback.print_exc()
            break  # 每个机型只跑一个 side


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="retarget pkl 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单文件实时查看
  python examples/viz_retarget_pkl.py \\
    --pkl /Users/wangduo/Downloads/retarget/example/vector_retargeting/output/inspire/right/0.pkl \\
    --hand inspire_hand --side right

  # 单文件保存 rrd
  python examples/viz_retarget_pkl.py \\
    --pkl .../inspire/right/0.pkl --hand inspire_hand --side right --save

  # 批量保存某机型所有 pkl
  python examples/viz_retarget_pkl.py \\
    --batch .../inspire/right/ --hand inspire_hand --side right --save

  # 一键跑所有机型第0个pkl
  python examples/viz_retarget_pkl.py \\
    --all --data-root .../output/ --save
        """
    )
    parser.add_argument("--pkl",  type=Path, help="单个 pkl 文件路径")
    parser.add_argument("--hand", type=str,  help="手型 ID（如 inspire_hand）")
    parser.add_argument("--side", type=str,  default="right",
                        choices=["left", "right"], help="左/右手（默认 right）")
    parser.add_argument("--batch", type=Path, help="批量模式：pkl 目录路径")
    parser.add_argument("--all",   action="store_true", help="一键跑所有机型")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/wangduo/Downloads/retarget/example/vector_retargeting/output"),
                        help="retarget 数据根目录（--all 模式使用）")
    parser.add_argument("--save",  action="store_true", help="保存为 .rrd 而非实时显示")
    parser.add_argument("--fps",   type=float, default=30.0, help="帧率（默认 30）")
    parser.add_argument("--max",   type=int,   default=None, help="批量模式最多处理几个文件")
    parser.add_argument("--idx",   type=int,   default=0,    help="--all 模式取第几个 pkl（默认 0）")

    args = parser.parse_args()
    output_dir = Path(__file__).parent.parent / "output" / "retarget"

    if args.all:
        viz_all_hands(args.data_root, save=True,
                      output_dir=output_dir, fps=args.fps, pkl_index=args.idx)

    elif args.batch:
        if not args.hand:
            parser.error("--batch 模式需要指定 --hand")
        viz_batch(args.batch, args.hand, args.side,
                  save=args.save, output_dir=output_dir,
                  fps=args.fps, max_files=args.max)

    elif args.pkl:
        if not args.hand:
            # 尝试从路径自动推断
            parts = args.pkl.parts
            for part in parts:
                if part in HAND_MAP:
                    args.hand = HAND_MAP[part]
                    print(f"[自动推断] hand={args.hand}")
                    break
        if not args.hand:
            parser.error("无法自动推断手型，请指定 --hand")
        viz_single_pkl(args.pkl, args.hand, args.side,
                       save=args.save, output_dir=output_dir, fps=args.fps)

    else:
        # 默认：跑 inspire/right/0.pkl 作为演示
        default_pkl = args.data_root / "inspire" / "right" / "0.pkl"
        print(f"[默认演示] {default_pkl}")
        viz_single_pkl(default_pkl, "inspire_hand", "right",
                       save=args.save, output_dir=output_dir, fps=args.fps)
