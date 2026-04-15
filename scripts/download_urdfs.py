#!/usr/bin/env python3
"""
URDF 文件下载辅助脚本

从 GitHub 下载各灵巧手的 URDF 和 mesh 文件到 urdf_assets/ 目录。

用法:
    python scripts/download_urdfs.py --hand unitree_dex3
    python scripts/download_urdfs.py --hand all
    python scripts/download_urdfs.py --list
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent
URDF_ASSETS = ROOT / "urdf_assets"

# 各手型的 GitHub 下载配置
# 格式: {hand_name: {"base_url": ..., "files": [相对路径, ...]}}
DOWNLOAD_CONFIGS = {
    "unitree_dex3": {
        "description": "宇树 Dex3-1（三指，7 DOF）",
        "base_url": (
            "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/"
            "master/robots/dexterous_hand_description/dex3_1"
        ),
        "files": [
            "dex3_1_l.urdf",
            "dex3_1_r.urdf",
            "meshes/left_hand_palm_link.STL",
            "meshes/left_hand_thumb_0_link.STL",
            "meshes/left_hand_thumb_1_link.STL",
            "meshes/left_hand_thumb_2_link.STL",
            "meshes/left_hand_middle_0_link.STL",
            "meshes/left_hand_middle_1_link.STL",
            "meshes/left_hand_index_0_link.STL",
            "meshes/left_hand_index_1_link.STL",
            "meshes/right_hand_palm_link.STL",
            "meshes/right_hand_thumb_0_link.STL",
            "meshes/right_hand_thumb_1_link.STL",
            "meshes/right_hand_thumb_2_link.STL",
            "meshes/right_hand_middle_0_link.STL",
            "meshes/right_hand_middle_1_link.STL",
            "meshes/right_hand_index_0_link.STL",
            "meshes/right_hand_index_1_link.STL",
        ],
        "target_dir": "unitree_dex3",
    },
    "unitree_dex1_1": {
        "description": "宇树 Dex1-1",
        "base_url": (
            "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/"
            "master/robots/dexterous_hand_description/dex1_1"
        ),
        "files": [
            "dex1_1.urdf",
            "meshes/base_link.STL",
            "meshes/Link1_1.STL",
            "meshes/Link1_2.STL",
            "meshes/Link1_3.STL",
            "meshes/Link2_1.STL",
            "meshes/Link2_2.STL",
            "meshes/Link2_3.STL",
        ],
        "target_dir": "unitree_dex1_1",
    },
    "inspire_dfq": {
        "description": "因时 Inspire DFQ（五指，15 DOF）",
        "base_url": (
            "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/"
            "master/robots/g1_description/inspire_hand"
        ),
        "files": [
            "DFQ_left_hand.urdf",
            "DFQ_right_hand.urdf",
            # mesh 文件较多，需要手动下载或 git clone
        ],
        "target_dir": "inspire_dfq",
        "note": "mesh 文件较多，建议使用 git clone 方式下载完整仓库",
    },
    "unitree_dex5": {
        "description": "宇树 Dex5-1（五指，16 DOF）",
        "base_url": None,
        "files": [],
        "target_dir": "unitree_dex5",
        "note": "Dex5-1 URDF 尚未公开，请关注 https://github.com/unitreerobotics/unitree_ros",
        "status": "unavailable",
    },
    "brainco": {
        "description": "强脑 BrainCo（五指，6 DOF）",
        "base_url": None,
        "files": [],
        "target_dir": "brainco",
        "note": "BrainCo URDF 尚未公开，请联系厂商获取",
        "status": "unavailable",
    },
    "sharpa": {
        "description": "Sharpa 灵巧手",
        "base_url": None,
        "files": [],
        "target_dir": "sharpa",
        "note": "Sharpa URDF 尚未公开，请联系厂商获取",
        "status": "unavailable",
    },
}


def download_file(url: str, dest: Path, verbose: bool = True) -> bool:
    """下载单个文件"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if verbose:
            print(f"  [跳过] 已存在: {dest.name}")
        return True
    try:
        if verbose:
            print(f"  [下载] {dest.name} ...", end="", flush=True)
        urllib.request.urlretrieve(url, dest)
        if verbose:
            size_kb = dest.stat().st_size / 1024
            print(f" {size_kb:.1f} KB ✓")
        return True
    except Exception as e:
        if verbose:
            print(f" ✗ 失败: {e}")
        return False


def download_hand(hand_name: str, verbose: bool = True) -> bool:
    """下载指定手型的 URDF 文件"""
    if hand_name not in DOWNLOAD_CONFIGS:
        print(f"[错误] 未知手型: {hand_name}")
        return False

    cfg = DOWNLOAD_CONFIGS[hand_name]
    print(f"\n{'='*55}")
    print(f"手型: {cfg['description']}")

    if cfg.get("status") == "unavailable":
        print(f"[不可用] {cfg.get('note', '暂无下载源')}")
        return False

    if not cfg.get("base_url") or not cfg.get("files"):
        print(f"[跳过] 无可下载文件")
        if cfg.get("note"):
            print(f"  提示: {cfg['note']}")
        return False

    target_dir = URDF_ASSETS / cfg["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for rel_path in cfg["files"]:
        url = f"{cfg['base_url']}/{rel_path}"
        dest = target_dir / rel_path
        if download_file(url, dest, verbose=verbose):
            success_count += 1

    total = len(cfg["files"])
    print(f"\n[结果] {success_count}/{total} 个文件下载成功")
    print(f"[目录] {target_dir}")

    if cfg.get("note"):
        print(f"[提示] {cfg['note']}")

    return success_count == total


def cmd_list():
    """列出所有可下载的手型"""
    print(f"\n{'='*65}")
    print(f"{'手型 ID':<20}  {'描述':<25}  {'状态'}")
    print(f"{'-'*65}")
    for hand_name, cfg in DOWNLOAD_CONFIGS.items():
        target_dir = URDF_ASSETS / cfg["target_dir"]
        urdf_files = list(target_dir.glob("*.urdf")) if target_dir.exists() else []
        local_status = f"已下载({len(urdf_files)}个URDF)" if urdf_files else "未下载"

        if cfg.get("status") == "unavailable":
            avail = "暂不可用"
        elif cfg.get("base_url"):
            avail = f"可下载({len(cfg['files'])}个文件)"
        else:
            avail = "无下载源"

        print(f"{hand_name:<20}  {cfg['description']:<25}  {avail} | 本地:{local_status}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser(description="灵巧手 URDF 下载工具")
    parser.add_argument(
        "--hand",
        type=str,
        default=None,
        help="手型 ID，或 'all' 下载所有可用手型",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有手型及下载状态",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    args = parser.parse_args()

    if args.list or args.hand is None:
        cmd_list()
        if args.hand is None:
            return

    if args.hand == "all":
        for hand_name in DOWNLOAD_CONFIGS:
            download_hand(hand_name, verbose=not args.quiet)
    else:
        download_hand(args.hand, verbose=not args.quiet)


if __name__ == "__main__":
    main()
