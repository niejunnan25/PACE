"""Reader for the LeRobot v2 episode layout used in the RoboTwin experiments."""

import json
from pathlib import Path

import numpy as np


def read_lerobot_episodes(dataset: Path, count: int | None = None):
    import pandas as pd

    info = json.loads((dataset / "meta/info.json").read_text())
    total = int(info["total_episodes"])
    count = total if count is None else count
    if not 1 <= count <= total:
        raise ValueError(f"episode count must be between 1 and {total}")
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    chunk_size = int(info.get("chunks_size", 1000))
    for episode in range(count):
        path = dataset / pattern.format(episode_chunk=episode // chunk_size, episode_index=episode)
        frame = pd.read_parquet(path, columns=["observation.state"])
        # Series.tolist preserves the dtype of the stored per-row arrays.
        yield np.array(frame["observation.state"].tolist())
