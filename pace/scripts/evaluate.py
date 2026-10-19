"""Evaluate 100 episodes of one RoboTwin task against an existing OpenPI server."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

PACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACE_ROOT / "src"))
ROBOTWIN_ROOT = PACE_ROOT.parent
sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(ROBOTWIN_ROOT / "description/utils"))

from pace import PaceConfig, select_horizon
from pace.robotwin import (
    load_cached_scenes,
    load_environment,
    load_tasks,
    restore_cached_scene_state,
    sample_instruction,
)
from pace.rollout import run_episode

PREDICTION_HORIZON = 50
INSTRUCTION_TYPE = "unseen"
UPSTREAM_COMMIT = "13c3c47ff4312dd62484bcd51be034af55c062d1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(load_tasks()), metavar="TASK")
    parser.add_argument("--method", choices=("pace", "fixed"), required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index for simulation")
    parser.add_argument("--horizon", type=int, help="Required for fixed; omit for PACE")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    # The batch worker supplies its run directory; standalone runs use a timestamp.
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-id", help="Optional operator-supplied checkpoint identity")
    parser.add_argument(
        "--ultrafast", action="store_true", help="Use the released expert-screened scene cache"
    )
    parser.add_argument(
        "--save-actions", action="store_true", help="Save full precision action chunks"
    )
    args = parser.parse_args(argv)
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("gpu must be nonnegative and port must be between 1 and 65535")
    if args.method == "pace":
        if args.horizon is not None:
            parser.error("PACE uses the paper settings; omit --horizon")
        args.horizon = PREDICTION_HORIZON
    elif args.horizon is None or not 1 <= args.horizon <= PREDICTION_HORIZON:
        parser.error(f"fixed requires --horizon between 1 and {PREDICTION_HORIZON}")
    args.episodes = 100
    if args.output is None:
        method = "pace" if args.method == "pace" else f"fixed_h{args.horizon}"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        args.output = PACE_ROOT / "runs" / stamp / args.task / method / f"seed{args.seed}"
    return args


def evaluate(args):
    config = PaceConfig()
    threshold_file = PACE_ROOT / "configs/calibration" / f"{args.task}.json"
    calibration = json.loads(threshold_file.read_text())
    cached_scenes = scene_cache = None
    if args.ultrafast:
        cached_scenes, scene_cache = load_cached_scenes(args.task, args.seed, args.episodes)

    def choose(actions):
        if len(actions) != PREDICTION_HORIZON:
            raise ValueError(
                f"the paper policy server must return {PREDICTION_HORIZON}-step action chunks"
            )
        if args.method == "fixed":
            return min(args.horizon, len(actions))
        return select_horizon(
            actions, calibration["threshold"], max_horizon=args.horizon, config=config
        )

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "task": args.task,
        "method": "pace" if args.method == "pace" else f"fixed_h{args.horizon}",
        "seed": args.seed,
        "expected_episodes": args.episodes,
        "max_horizon": args.horizon,
        "prediction_horizon": PREDICTION_HORIZON,
        "task_config": "demo_clean",
        "instruction_type": INSTRUCTION_TYPE,
        "expert_check_mode": "cached_scenes" if args.ultrafast else "online",
        "scene_cache": scene_cache,
        "checkpoint_id": args.checkpoint_id,
        "checkpoint_id_source": "operator declaration" if args.checkpoint_id else "not supplied",
        "gpu": args.gpu,
        "server_endpoint": f"{args.host}:{args.port}",
        "selector": asdict(config),
        "threshold": calibration["threshold"] if args.method == "pace" else None,
        "calibration_sha256": hashlib.sha256(threshold_file.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROBOTWIN_ROOT, text=True
        ).strip(),
        "code_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROBOTWIN_ROOT)
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "status": "running",
    }
    metadata_path = args.output / "metadata.json"
    write_metadata(metadata_path, metadata)
    print(
        f"[PACE] {args.task} | {metadata['method']} | seed={args.seed} | 100 episodes", flush=True
    )
    print(f"[PACE] Results: {args.output}", flush=True)
    print(f"[PACE] Expert screening: {metadata['expert_check_mode']}", flush=True)
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        policy = WebsocketClientPolicy(args.host, args.port)
        server_metadata = policy.get_server_metadata()
        expected = {
            "task": args.task,
            "config": f"pi05_robotwin_{args.task}",
            "action_horizon": PREDICTION_HORIZON,
            "action_dim": 14,
            "action_space": "absolute_joint",
        }
        if any(server_metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("server task/config/action metadata does not match the requested task")
        metadata["server_metadata"] = server_metadata
        write_metadata(metadata_path, metadata)
        result = collect_episodes(args, choose, metadata["instruction_type"], policy, cached_scenes)
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_metadata(metadata_path, metadata)
        raise
    metadata.update(status="complete", **result)
    write_metadata(metadata_path, metadata)


def write_metadata(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def collect_episodes(args, choose, instruction_type, policy, cached_scenes=None):
    from envs.utils.create_actor import UnStableError

    env, env_args = load_environment(args.task)
    env.suc = 0
    env.test_num = 0
    scene_seed = 100000 * (1 + args.seed)
    completed = 0
    successes = 0

    with (
        (args.output / "episodes.jsonl").open("w") as episodes,
        (args.output / "skipped_scenes.jsonl").open("w") as skipped,
    ):
        while completed < args.episodes:
            if cached_scenes is not None:
                scene = cached_scenes[completed]
                scene_seed = scene["scene_seed"]
                info = {"info": scene["episode_info"]}
            else:
                expert_args = {**env_args, "render_freq": 0}
                try:
                    env.setup_demo(
                        now_ep_num=completed, seed=scene_seed, is_test=True, **expert_args
                    )
                    info = env.play_once()
                except UnStableError:
                    skipped.write(
                        json.dumps({"scene_seed": scene_seed, "reason": "unstable"}) + "\n"
                    )
                    scene_seed += 1
                    continue
                except Exception as error:
                    # Upstream rejects candidates whose expert setup or plan raises.
                    detail = f"{type(error).__name__}: {error}"
                    skipped.write(
                        json.dumps(
                            {"scene_seed": scene_seed, "reason": "expert_error", "error": detail}
                        )
                        + "\n"
                    )
                    skipped.flush()
                    print(f"[PACE] Skipping expert scene {scene_seed}: {detail}", flush=True)
                    scene_seed += 1
                    continue
                finally:
                    env.close_env()
                if not (env.plan_success and env.check_success()):
                    skipped.write(
                        json.dumps({"scene_seed": scene_seed, "reason": "expert_failed"}) + "\n"
                    )
                    scene_seed += 1
                    continue

            query_records = []

            def record_query(step, actions, horizon):
                record = {"step": step, "horizon": horizon, "chunk_length": len(actions)}
                if args.save_actions:
                    record["actions"] = actions.tolist()
                query_records.append(record)

            try:
                env.setup_demo(now_ep_num=completed, seed=scene_seed, is_test=True, **env_args)
                if cached_scenes is not None:
                    restore_cached_scene_state(args.task, env, info["info"])
                instruction = sample_instruction(args.task, info, args.episodes, instruction_type)
                env.set_instruction(instruction)
                print(
                    f"[PACE] {args.task} | episode {completed + 1}/100 START | scene seed={scene_seed}",
                    flush=True,
                )
                result = run_episode(env, policy, choose, record_query)
            finally:
                # Match the original cleanup cadence after accepted episode k.
                env.close_env(clear_cache=((completed + 2) % env_args["clear_cache_freq"] == 0))
            record = {
                "episode": completed,
                "scene_seed": scene_seed,
                "instruction": instruction,
                **result,
                "termination": "success" if result["success"] else "step_limit",
            }
            episodes.write(json.dumps(record) + "\n")
            episodes.flush()
            (args.output / f"episode_{completed:03d}_queries.json").write_text(
                json.dumps(query_records) + "\n"
            )
            outcome = "SUCCESS" if result["success"] else "FAIL"
            (args.output / f"episode_{completed:03d}_{outcome.lower()}.log").write_text(
                f"Task: {args.task}\nEpisode: {completed + 1}/100\nScene seed: {scene_seed}\n"
                f"Instruction: {instruction}\nResult: {outcome}\n"
                f"Termination: {record['termination']}\nSteps: {result['steps']}\n"
                f"Queries: {result['queries']}\n"
                + "".join(
                    f"Query {i + 1}: step={q['step']}, horizon={q['horizon']}\n"
                    for i, q in enumerate(query_records)
                )
            )
            completed += 1
            successes += int(result["success"])
            env.test_num = completed
            env.suc = successes
            print(
                f"[PACE] {args.task} | episode {completed}/100 {outcome} | "
                f"success {successes}/{completed} ({100 * successes / completed:.1f}%) | "
                f"scene seed={scene_seed} | steps={result['steps']} queries={result['queries']}",
                flush=True,
            )
            scene_seed += 1

    return {
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / completed,
    }


def handle_termination(signum, _frame):
    raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")


def main():
    signal.signal(signal.SIGTERM, handle_termination)
    arguments = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu)
    arguments.output = arguments.output.resolve()
    os.chdir(ROBOTWIN_ROOT)
    evaluate(arguments)


if __name__ == "__main__":
    main()
