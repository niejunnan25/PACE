"""Recompute a task threshold from its existing LeRobot demonstrations."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACE_ROOT / "src"))

from pace import PaceConfig
from pace.calibration import calibrate
from pace.datasets import read_lerobot_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.dataset / "meta/info.json").read_text())
    config = PaceConfig(fps=float(info["fps"]))
    stats = calibrate(read_lerobot_episodes(args.dataset, args.episodes), config=config)
    stats.update(dataset_id=args.dataset.name, selector=asdict(config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Recalibration must not silently replace a published threshold.
    with args.output.open("x") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(
        f"p5 threshold: {stats['threshold']:.12g}; {stats['num_episodes_analyzed']} demonstrations"
    )


if __name__ == "__main__":
    main()
