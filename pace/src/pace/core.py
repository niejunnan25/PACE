"""Joint-space PACE, independent of the policy and simulator.

The returned horizon is an action count. Valley indices follow the original
RoboTwin implementation: a valley at speed index k requests k actions, without
adding one for the finite difference. Each candidate is measured against the
start of the current chunk, including candidates after a rejected valley.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

ALOHA_ARMS = (slice(0, 6), slice(7, 13))


@dataclass(frozen=True)
class PaceConfig:
    fps: float = 50.0
    min_distance: int = 10
    smoothing_window: int = 11
    polynomial_order: int = 3

    def __post_init__(self):
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if self.min_distance < 1:
            raise ValueError("min_distance must be positive")
        if self.smoothing_window < 1 or self.smoothing_window % 2 != 1:
            raise ValueError("smoothing_window must be a positive odd integer")
        if not 0 <= self.polynomial_order < self.smoothing_window:
            raise ValueError("polynomial_order must be smaller than smoothing_window")


def smooth_speed(speed: np.ndarray, config: PaceConfig) -> np.ndarray:
    """Use the largest supported odd window, up to the configured width."""
    window = min(config.smoothing_window, (len(speed) - 1) // 2 * 2 + 1)
    if window < max(5, config.polynomial_order + 1):
        return speed.copy()
    return savgol_filter(speed, window, config.polynomial_order)


def find_valleys(smoothed: np.ndarray, min_distance: int) -> np.ndarray:
    if len(smoothed) < min_distance:
        return np.empty(0, dtype=int)
    return find_peaks(-smoothed, distance=min_distance)[0]


def segment_contrast(speed: np.ndarray, start: int, end: int) -> float:
    peak = float(np.max(speed[start : end + 1]))
    return min(peak - float(speed[start]), peak - float(speed[end]))


def arm_speeds(
    positions: np.ndarray,
    fps: float,
    arms: tuple[slice, ...] = ALOHA_ARMS,
) -> list[np.ndarray]:
    return [np.linalg.norm(np.diff(positions[:, arm], axis=0) * fps, axis=1) for arm in arms]


def validate_arms(arms: tuple[slice, ...], channels: int) -> None:
    if not arms or any(
        arm.step not in (None, 1)
        or not 0 <= (arm.start or 0) < (channels if arm.stop is None else arm.stop) <= channels
        for arm in arms
    ):
        raise ValueError("arms must select nonempty joint groups within the channel dimension")


def select_horizon(
    actions: np.ndarray,
    threshold: float,
    *,
    max_horizon: int = 50,
    config: PaceConfig = PaceConfig(),
    arms: tuple[slice, ...] = ALOHA_ARMS,
) -> int:
    """Select the earliest accepted valley across arms, or the available horizon.

    actions: (time, channels) absolute, decoded joint commands. Gripper channels
        must be excluded from arms. Defaults match RoboTwin's 14-D ALOHA layout.
    threshold: absolute speed contrast calibrated from training demonstrations.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("actions must be a nonempty, finite (time, channels) array")
    validate_arms(arms, actions.shape[1])
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not isinstance(max_horizon, (int, np.integer)) or max_horizon < 1:
        raise ValueError("max_horizon must be a positive integer action count")

    horizon = min(max_horizon, len(actions))
    for speed in arm_speeds(actions, config.fps, arms):
        smoothed = smooth_speed(speed, config)
        for valley in find_valleys(smoothed, config.min_distance):
            if valley > horizon:
                break
            if segment_contrast(smoothed, 0, int(valley)) >= threshold:
                horizon = min(horizon, int(valley))
                break
    return int(horizon)
