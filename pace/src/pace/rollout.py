"""One execution loop shared by PACE and fixed-horizon evaluation."""

from collections.abc import Callable

import numpy as np


def encode_observation(observation: dict, instruction: str) -> dict:
    cameras = observation["observation"]
    return {
        "state": observation["joint_action"]["vector"],
        "images": {
            "cam_high": cameras["head_camera"]["rgb"].transpose(2, 0, 1),
            "cam_left_wrist": cameras["left_camera"]["rgb"].transpose(2, 0, 1),
            "cam_right_wrist": cameras["right_camera"]["rgb"].transpose(2, 0, 1),
        },
        "prompt": instruction,
    }


def run_episode(env, policy, choose_horizon: Callable[[np.ndarray], int], on_query=None) -> dict:
    """Query only at execution boundaries; discard the unexecuted chunk suffix.

    policy implements infer(observation) -> {"actions": (T, 14)}. Errors propagate
    to the caller: a failed policy service must not become a task failure score.
    on_query optionally receives (env_step, actions, planned_horizon).
    """
    queries = 0
    while env.take_action_cnt < env.step_lim and not env.eval_success:
        response = policy.infer(encode_observation(env.get_obs(), env.get_instruction()))
        actions = np.asarray(response["actions"])
        if (
            actions.ndim != 2
            or actions.shape[1] != 14
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("policy must return a nonempty finite (T, 14) action chunk")
        horizon = choose_horizon(actions)
        if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= len(actions):
            raise ValueError("execution horizon must be an action count within the chunk")
        queries += 1
        if on_query is not None:
            on_query(int(env.take_action_cnt), actions, int(horizon))
        for action in actions[:horizon]:
            env.take_action(action)
            if env.eval_success or env.take_action_cnt >= env.step_lim:
                break
    return {
        "success": bool(env.eval_success),
        "steps": int(env.take_action_cnt),
        "queries": queries,
    }
