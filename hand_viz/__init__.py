"""
灵巧手 Rerun 可视化工具包

支持的手型:
  - unitree_dex1_1  : 宇树 Dex1-1
  - unitree_dex3    : 宇树 Dex3-1（三指，7 DOF）
  - unitree_dex5    : 宇树 Dex5-1（五指，16 DOF）
  - inspire_dfq     : 因时 Inspire DFQ（五指，15 DOF）
  - brainco         : 强脑 BrainCo（五指，6 DOF）
  - sharpa          : Sharpa 灵巧手

快速使用:
    from hand_viz import HandVisualizer
    viz = HandVisualizer("unitree_dex3")
    viz.visualize(joint_angles_sequence, side="left")
"""

from .visualizer import HandVisualizer
from .config_loader import HandConfig, load_hand_config, list_available_hands

__all__ = ["HandVisualizer", "HandConfig", "load_hand_config", "list_available_hands"]
__version__ = "0.1.0"
