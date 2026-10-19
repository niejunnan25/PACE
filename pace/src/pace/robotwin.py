"""RoboTwin stable_2.0 environment and paper protocol adapter."""

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

PACE_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = PACE_ROOT.parent


def load_tasks():
    """Use the same task list as the batch scheduler."""
    return [
        task for task in (PACE_ROOT / "configs/task_order.txt").read_text().splitlines() if task
    ]


def load_environment(task: str):
    """Keep upstream files intact; apply the paper's observation flags in memory."""
    args = yaml.safe_load((ROBOTWIN_ROOT / "task_config/demo_clean.yml").read_text())
    embodiments = yaml.safe_load((ROBOTWIN_ROOT / "task_config/_embodiment_config.yml").read_text())
    cameras = yaml.safe_load((ROBOTWIN_ROOT / "task_config/_camera_config.yml").read_text())
    camera = cameras[args["camera"]["head_camera_type"]]
    robot_path = Path(embodiments[args["embodiment"][0]]["file_path"])
    if not robot_path.is_absolute():
        robot_path = ROBOTWIN_ROOT / robot_path
    robot = yaml.safe_load((robot_path / "config.yml").read_text())
    args.update(
        task_name=task,
        task_config="demo_clean",
        head_camera_h=camera["h"],
        head_camera_w=camera["w"],
        left_robot_file=str(robot_path),
        right_robot_file=str(robot_path),
        left_embodiment_config=robot,
        right_embodiment_config=deepcopy(robot),
        dual_arm_embodied=True,
        eval_mode=True,
        eval_video_log=False,
    )
    args["data_type"]["endpose"] = False
    module = importlib.import_module(f"envs.{task}")
    return getattr(module, task)(), args


def sample_instruction(task: str, episode_info: dict, count: int, split: str) -> str:
    from generate_episode_instructions import generate_episode_descriptions

    descriptions = generate_episode_descriptions(task, [episode_info["info"]], count)
    return str(np.random.choice(descriptions[0][split]))


def load_cached_scenes(task: str, seed: int, count: int):
    """Load the released, ordered expert-accepted scenes for one evaluation seed."""
    path = PACE_ROOT / "configs/expert_scenes.json"
    content = path.read_bytes()
    cache = json.loads(content)
    # Asset locations in _embodiment_config.yml are local to each workspace.
    for name in (
        f"envs/{task}.py",
        "envs/_base_task.py",
        "task_config/demo_clean.yml",
        "task_config/_camera_config.yml",
    ):
        if (
            hashlib.sha256((ROBOTWIN_ROOT / name).read_bytes()).hexdigest()
            != cache["compatible_source_files_sha256"][name]
        ):
            raise ValueError(
                f"scene cache does not match {name}; omit --ultrafast for online screening"
            )
    scenes = cache["tasks"][task][str(seed)]
    if len(scenes) < count:
        raise ValueError(f"scene cache has only {len(scenes)} episodes for {task}, seed {seed}")
    return scenes[:count], {"file": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def restore_cached_scene_state(task: str, env, episode_info: dict):
    """Set fields normally initialized by expert play_once(), before policy execution."""
    if task in {"open_laptop", "place_object_scale", "put_object_cabinet"}:
        env.arm_tag = episode_info["{a}"]
    if task == "put_object_cabinet":
        env.origin_z = float(env.object.get_pose().p[2])
