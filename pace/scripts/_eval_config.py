"""Read the three paths in eval.yaml for the Bash batch scheduler."""

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit(
        "ERROR: PyYAML is missing. Activate the RoboTwin environment and install './pace[robotwin]'."
    ) from None


def read_paths(robotwin_root):
    config_path = Path(__file__).with_name("eval.yaml")
    config = yaml.safe_load(config_path.read_text())
    fields = ("openpi_root", "checkpoint_root", "server_python")
    if not isinstance(config, dict) or set(config) != set(fields):
        raise ValueError(f"expected these fields: {', '.join(fields)}")

    def path(field, base):
        value = config[field]
        if not isinstance(value, str) or not value.strip() or any(c in value for c in "\n\r\0"):
            raise ValueError(f"{field} must be a nonempty, single-line path")
        # abspath keeps interpreter symlinks intact, preserving environment selection.
        return Path(os.path.abspath(base / Path(value).expanduser()))

    openpi = path("openpi_root", robotwin_root)
    return openpi, path("checkpoint_root", openpi), path("server_python", openpi)


if __name__ == "__main__":
    try:
        paths = read_paths(Path(sys.argv[1]))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(f"ERROR: pace/scripts/eval.yaml: {error}") from None
    for configured_path in paths:
        print(configured_path)
