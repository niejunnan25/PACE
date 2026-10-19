"""Select an execution horizon for a synthetic joint-action chunk."""

import sys
from pathlib import Path

import numpy as np

PACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACE_ROOT / "src"))

from pace import select_horizon

# Two smooth motion segments separated by a slow point, sampled at 50 Hz.
joint_speed = 0.05 + np.sin(np.pi * np.arange(49) / 24) ** 2
actions = np.zeros((50, 14))
actions[1:, 0] = np.cumsum(joint_speed) / 50

# Illustrative only; use a calibrated task threshold for a real policy.
horizon = select_horizon(actions, threshold=0.5)
print(f"Synthetic example: execute {horizon} of {len(actions)} actions")
