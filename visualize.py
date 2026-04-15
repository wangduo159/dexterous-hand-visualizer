#!/usr/bin/env python3
"""
灵巧手可视化主入口脚本

用法:
    # 可视化 numpy 文件中的关节角度序列
    python visualize.py --hand unitree_dex3 --data path/to/angles.npy --side left

    # 可视化 JSON 文件
    python visualize.py --hand inspire_dfq --data path/to/angles.json --fps 25

    # 保存为 .rrd 文件（不启动 Viewer）
    python visualize.py --hand unitree_dex3 --data angles.npy --save output.rrd

    # 运行内置演示（无需数据文件）
    python visualize.py --hand unitree_dex3 --demo

    # 列出所有支持的手型
    python visualize.py --list

    # 打印某手型的关节信息
    python visualize.py --hand unitree_dex3 --info
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cmd_list_hands():
    """列出所有支持的手型"""
    from hand_viz import list_available_hands
    hands = list_available_hands()
    print(f"\n{'='*65}")
    print(f"{'手型 ID':<20}  {'显示名称':<20}  {'DOF':>5}  {'URDF状态'}")
    print(f"{'-'*65}")
    for hand_id, info in hands.items():
        if "error" in info:
            print(f"{hand_id:<20}  {'[配置加载失败]':<20}  {'?':>5}  {info['error']}")
        else:
            left_ok  = "✓" if info["urdf_ready_left"]  else "✗"
            right_ok = "✓" if info["urdf_ready_right"] else "✗"
            print(
                f"{hand_id:<20}  {info['display_name']:<20}  {info['dof']:>5}  "
                f"左:{left_ok} 右:{right_ok}"
            )
    print(f"{'='*65}")
    print("\n提示: ✗ 表示 URDF 文件尚未下载，请参考 README.md 获取下载方法\n")


def cmd_info(hand_name: str, side: str = "left"):
    """打印手型关节信息"""
    from hand_viz import HandVisualizer
    viz = HandVisualizer(hand_name, spawn_viewer=False)
    viz.print_joint_info(side=side)


def cmd_demo(hand_name: str, side: str = "left", fps: float = 30.0, save_path: str = None):
    """
    运行内置演示：生成正弦波关节角度序列并可视化
    """
    from hand_viz import HandVisualizer, load_hand_config

    config = load_hand_config(hand_name)
    dof = config.dof
    T = int(fps * 5)  # 5 秒演示

    print(f"[演示] 手型: {config.display_name}，DOF={dof}，{T}帧（{T/fps:.1f}s）")

    # 生成正弦波序列：每个关节相位不同
    t = np.linspace(0, 2 * np.pi, T)
    joints_cfg = config.joints.get(side) or config.joints.get("default", [])
    joints_cfg = sorted(joints_cfg, key=lambda j: j.dof_index)

    seq = np.zeros((T, dof))
    for i, joint in enumerate(joints_cfg):
        amplitude = (joint.limit_upper - joint.limit_lower) * 0.4
        center = (joint.limit_upper + joint.limit_lower) / 2.0
        phase = i * (2 * np.pi / max(dof, 1))
        seq[:, i] = center + amplitude * np.sin(t + phase)

    viz = HandVisualizer(hand_name, spawn_viewer=(save_path is None))
    if save_path:
        viz.save_rrd(seq, save_path, side=side, fps=fps)
    else:
        viz.visualize(seq, side=side, fps=fps)


def cmd_visualize(
    hand_name: str,
    data_path: str,
    side: str = "left",
    fps: float = 30.0,
    save_path: str = None,
    bimanual: bool = False,
    right_data_path: str = None,
):
    """从文件加载数据并可视化"""
    from hand_viz import HandVisualizer

    def load_data(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            print(f"[错误] 数据文件不存在: {path}")
            sys.exit(1)
        if p.suffix == ".npy":
            return np.load(path)
        elif p.suffix == ".npz":
            data = np.load(path)
            # 取第一个数组
            key = list(data.keys())[0]
            print(f"[提示] 从 .npz 中读取键: '{key}'")
            return data[key]
        elif p.suffix in (".json", ".jsonl"):
            with open(path) as f:
                raw = json.load(f)
            return np.array(raw)
        elif p.suffix == ".csv":
            return np.loadtxt(path, delimiter=",")
        else:
            print(f"[错误] 不支持的文件格式: {p.suffix}（支持: .npy .npz .json .csv）")
            sys.exit(1)

    angles = load_data(data_path)
    if angles.ndim == 1:
        angles = angles[np.newaxis, :]  # 单帧 → (1, dof)

    print(f"[数据] shape={angles.shape}，dtype={angles.dtype}")

    viz = HandVisualizer(hand_name, spawn_viewer=(save_path is None))

    if bimanual and right_data_path:
        right_angles = load_data(right_data_path)
        if right_angles.ndim == 1:
            right_angles = right_angles[np.newaxis, :]
        viz.visualize_bimanual(angles, right_angles, fps=fps, save_path=save_path)
    else:
        viz.visualize(angles, side=side, fps=fps, save_path=save_path)


def main():
    parser = argparse.ArgumentParser(
        description="灵巧手 Rerun 可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--hand", "-H",
        type=str,
        default="unitree_dex3",
        help="手型 ID（默认: unitree_dex3）",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="关节角度数据文件路径（.npy / .npz / .json / .csv）",
    )
    parser.add_argument(
        "--right-data",
        type=str,
        default=None,
        help="右手数据文件路径（双手模式）",
    )
    parser.add_argument(
        "--side", "-s",
        type=str,
        default="left",
        choices=["left", "right"],
        help="手的侧别（默认: left）",
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=30.0,
        help="帧率（默认: 30.0）",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="保存为 .rrd 文件路径（不指定则实时显示）",
    )
    parser.add_argument(
        "--bimanual",
        action="store_true",
        help="双手模式（需同时提供 --data 和 --right-data）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行内置演示（正弦波动画，无需数据文件）",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有支持的手型",
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="打印指定手型的关节信息",
    )

    args = parser.parse_args()

    # 确保 hand_viz 包可以被导入
    sys.path.insert(0, str(Path(__file__).parent))

    if args.list:
        cmd_list_hands()
        return

    if args.info:
        cmd_info(args.hand, side=args.side)
        return

    if args.demo:
        cmd_demo(args.hand, side=args.side, fps=args.fps, save_path=args.save)
        return

    if args.data is None:
        print("[错误] 请提供数据文件路径（--data）或使用 --demo 运行演示")
        parser.print_help()
        sys.exit(1)

    cmd_visualize(
        hand_name=args.hand,
        data_path=args.data,
        side=args.side,
        fps=args.fps,
        save_path=args.save,
        bimanual=args.bimanual,
        right_data_path=args.right_data,
    )


if __name__ == "__main__":
    main()
