# 🤖 Dexterous Hand Visualizer

A multi-brand dexterous hand visualization framework powered by [Rerun](https://rerun.io).  
Feed it a hand model name + a joint-angle sequence, and watch the 3-D animation play back in the Rerun Viewer — or save it as a portable `.rrd` file.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Rerun](https://img.shields.io/badge/rerun--sdk-0.21%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **12 hand models out of the box** — Unitree Dex3-1 / Dex1-1 / Dex5-1, Inspire Hand, BrainCo Revo2, Sharpa Wave, Allegro, Shadow, LEAP, SVH, Ability Hand, Panda Gripper
- **YAML-driven configuration** — add a new hand by dropping in one config file, no code changes needed
- **Bimanual support** — visualize left and right hands simultaneously with a split-panel layout
- **Multiple input formats** — `.npy`, `.npz`, `.json`, `.csv`
- **dex-retargeting integration** — built-in joint-order alignment helpers for [dex-retargeting](https://github.com/dexsuite/dex-retargeting) output
- **Offline playback** — save recordings as `.rrd` files and open them later with `rerun <file>.rrd`
- **One-command URDF download** — fetch URDF + mesh assets directly from upstream repos

---

## 🦾 Supported Hand Models

| Hand ID | Display Name | DOF | URDF |
|---|---|---|---|
| `unitree_dex3` | Unitree Dex3-1 | 7 | ✅ auto-download |
| `unitree_dex1_1` | Unitree Dex1-1 | 6 | ✅ auto-download |
| `unitree_dex5_1` | Unitree Dex5-1 | 16 | ⏳ pending release |
| `inspire_hand` | Inspire Hand | 12 | ✅ auto-download |
| `brainco_revo2` | BrainCo Revo2 | 10 | ✅ bundled |
| `sharpa_wave` | Sharpa Wave | 12 | ✅ bundled |
| `allegro_hand` | Allegro Hand | 16 | via dex-retargeting |
| `shadow_hand` | Shadow Hand | 24 | via dex-retargeting |
| `leap_hand` | LEAP Hand | 16 | via dex-retargeting |
| `svh_hand` | SVH Hand | 9 | via dex-retargeting |
| `ability_hand` | Ability Hand | 10 | via dex-retargeting |
| `panda_gripper` | Panda Gripper | 2 | via dex-retargeting |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download URDF assets

```bash
# Download Unitree Dex3-1 (recommended starting point)
python scripts/download_urdfs.py --hand unitree_dex3

# Download all available hands at once
python scripts/download_urdfs.py --hand all

# Check download status
python scripts/download_urdfs.py --list
```

### 3. Run the built-in demo

```bash
# Sine-wave animation — no data file needed
python visualize.py --hand unitree_dex3 --demo

# Print joint info (verify DOF order before feeding your own data)
python visualize.py --hand unitree_dex3 --info

# List all registered hands
python visualize.py --list
```

### 4. Visualize your own data

```bash
# From a .npy file  (shape: T × DOF, unit: radians)
python visualize.py --hand unitree_dex3 --data your_angles.npy --side left

# Custom frame rate
python visualize.py --hand unitree_dex3 --data angles.npy --fps 25

# Save to .rrd (open later with: rerun output.rrd)
python visualize.py --hand unitree_dex3 --data angles.npy --save output.rrd

# Bimanual mode
python visualize.py --hand unitree_dex3 --bimanual \
    --data left_angles.npy --right-data right_angles.npy
```

---

## 🐍 Python API

```python
import numpy as np
from hand_viz import HandVisualizer

# Create a visualizer
viz = HandVisualizer("unitree_dex3")

# Inspect joint order before feeding data
viz.print_joint_info(side="left")

# Visualize a sequence  (T frames × DOF)
seq = np.load("retarget_output.npy")   # shape=(T, 7)
viz.visualize(seq, side="left", fps=30.0)

# Bimanual visualization
viz.visualize_bimanual(left_seq, right_seq, fps=30.0)

# Save as .rrd without opening the viewer
viz.save_rrd(seq, "output.rrd", side="left", fps=30.0)
```

---

## 🔗 Integration with dex-retargeting

```python
from hand_viz import load_hand_config
from examples.demo_retarget_data import align_retarget_to_config

# dex-retargeting output
retarget_joint_names = retargeting.joint_names        # list[str]
retarget_angles      = retargeting.retarget(positions) # shape=(dof,)

# Re-order joints to match this tool's config
aligned = align_retarget_to_config(
    retarget_joint_names,
    retarget_angles,
    hand_name="unitree_dex3",
    side="left",
)

# Visualize
viz = HandVisualizer("unitree_dex3")
viz.visualize(aligned, side="left")
```

---

## 📁 Project Structure

```
.
├── visualize.py              # CLI entry point
├── requirements.txt
├── hand_viz/                 # Core library
│   ├── __init__.py
│   ├── config_loader.py      # YAML config loader + hand registry
│   ├── urdf_loader.py        # URDF parsing + joint driving
│   └── visualizer.py        # Unified visualization API
├── configs/                  # Per-hand joint mapping configs
│   ├── unitree_dex3.yaml
│   ├── unitree_dex1_1.yaml
│   ├── inspire_hand.yaml
│   └── ...
├── urdf_assets/              # URDF + mesh files (downloaded separately)
│   ├── unitree_dex3/
│   ├── inspire_hand/
│   └── ...
├── examples/                 # Example scripts
│   ├── demo_dex3.py          # Dex3-1 grasp / wave demo
│   ├── demo_new_hands.py     # Multi-hand demo
│   ├── demo_retarget_data.py # dex-retargeting integration
│   └── demo_retarget_hands.py
└── scripts/
    └── download_urdfs.py     # URDF download helper
```

---

## 📐 Data Format

Joint angles are expected in **radians**, with shape `(T, DOF)` for sequences or `(DOF,)` for a single frame.

The joint order for each hand is defined by `dof_index` in `configs/*.yaml`.  
Run `python visualize.py --hand <id> --info` to print the exact order.

| Format | Notes |
|---|---|
| `.npy` | NumPy array — recommended |
| `.npz` | NumPy archive — first key is used |
| `.json` | JSON array |
| `.csv` | Comma-separated, one frame per row |

---

## ➕ Adding a New Hand

1. Create `configs/your_hand.yaml` (use `configs/unitree_dex3.yaml` as a template)
2. Register the hand in `hand_viz/config_loader.py` → `HAND_REGISTRY`
3. Place URDF + mesh files under `urdf_assets/your_hand/`
4. *(Optional)* Add a download entry in `scripts/download_urdfs.py`

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `rerun-sdk >= 0.21` | 3-D visualization engine |
| `numpy >= 1.24` | Numerical computation |
| `pyyaml >= 6.0` | Config file parsing |
| `yourdfpy >= 0.0.56` | URDF parsing (joint axis extraction) |

---

## 🗂️ URDF Sources

| Hand | Source |
|---|---|
| Unitree Dex1-1 / Dex3-1 | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) |
| Inspire Hand | [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) |
| Allegro / Shadow / LEAP / SVH / Ability / Panda | [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) |
| Unitree Dex5-1 | Pending official release |
| BrainCo Revo2 / Sharpa Wave | Contact manufacturer |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
