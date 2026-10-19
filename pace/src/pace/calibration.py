"""Calibrate the absolute contrast threshold from demonstration joint states."""

from collections.abc import Iterable

import numpy as np

from .core import (
    ALOHA_ARMS,
    PaceConfig,
    arm_speeds,
    find_valleys,
    segment_contrast,
    smooth_speed,
    validate_arms,
)


def calibrate(
    episodes: Iterable[np.ndarray],
    *,
    config: PaceConfig = PaceConfig(),
    percentile: float = 5.0,
    arms: tuple[slice, ...] = ALOHA_ARMS,
) -> dict:
    """Pool positive contrasts from consecutive valley-bounded segments.

    As in the paper implementation, select the more active arm per episode by
    mean *unsmoothed* speed (first arm wins a tie), include the two trajectory
    endpoints, and skip demonstrations shorter than 20 states. Preserve the
    input dtype: historical LeRobot states are float32, unlike online actions
    which the selector explicitly promotes to float64.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    contrasts = []
    analyzed = 0
    skipped_short = 0
    for states in episodes:
        states = np.asarray(states)
        if states.ndim != 2 or not np.isfinite(states).all():
            raise ValueError("each episode must be a finite (time, channels) array")
        validate_arms(arms, states.shape[1])
        if len(states) < 20:
            skipped_short += 1
            continue
        speeds = arm_speeds(states, config.fps, arms)
        active = speeds[int(np.argmax([np.mean(speed) for speed in speeds]))]
        smoothed = smooth_speed(active, config)
        valleys = find_valleys(smoothed, config.min_distance)
        boundaries = [0, *valleys, len(smoothed) - 1]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end > start:
                contrast = segment_contrast(smoothed, int(start), int(end))
                if contrast > 0:
                    contrasts.append(contrast)
        analyzed += 1

    if not contrasts:
        raise ValueError("demonstrations contain no positive segment contrasts")
    values = np.asarray(contrasts)
    return {
        "threshold": float(np.percentile(values, percentile)),
        "percentile": percentile,
        "num_episodes_analyzed": analyzed,
        "num_episodes_skipped_short": skipped_short,
        "num_segments_analyzed": len(values),
        "peak_height_mean": float(np.mean(values)),
        "peak_height_std": float(np.std(values)),
        "peak_height_min": float(np.min(values)),
        "peak_height_max": float(np.max(values)),
        **{f"peak_height_p{p}": float(np.percentile(values, p)) for p in (5, 10, 25, 50, 75, 90)},
    }
