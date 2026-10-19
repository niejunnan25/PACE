"""Process lifecycle for eval_all.sh; standalone evaluation uses evaluate.py."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def stop(process):
    if process is None:
        return
    # A worker may outlive its parent. Always clean up the entire process group.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def check_port(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Probe the same address used by the health check and evaluation client.
        probe.bind(("127.0.0.1", port))


def wait_ready(server, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("policy server exited; see server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.2)
    raise TimeoutError("policy server did not become ready; see server.log")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def copy_progress(stream, log, label):
    for line in stream:
        log.write(line)
        log.flush()
        if line.startswith("[PACE]"):
            print(f"{label} {line.rstrip()}", flush=True)


def run(args):
    # Keep the lock file: unlinking it would let concurrent workers lock different inodes.
    lock_path = Path("/tmp") / f"robotwin-pace-{os.getuid()}-port{args.port}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"port {args.port} is reserved by another PACE evaluation") from None
        run_on_port(args)


def run_on_port(args):
    checkpoint = (args.checkpoint_root / args.task).resolve()
    openpi = args.openpi_root.resolve()
    robotwin = args.robotwin_root.resolve()
    output = args.output.resolve()
    check_port(args.port)
    output.mkdir(parents=True, exist_ok=False)
    server = client = reader = None
    label = f"[GPU {args.gpu} port {args.port}]"
    launch = {
        "task": args.task,
        "method": args.method,
        "seed": args.seed,
        "gpu": args.gpu,
        "port": args.port,
        "checkpoint_dir": str(checkpoint),
        "expert_check_mode": "cached_scenes" if args.ultrafast else "online",
        "status": "starting",
        "created_utc": utc_now(),
    }
    try:
        launch["norm_stats_sha256"] = hashlib.sha256(
            (checkpoint / "assets/robotwin/norm_stats.json").read_bytes()
        ).hexdigest()
        server_command = [
            os.path.abspath(args.server_python),
            "-u",
            str(openpi / "scripts/serve_policy.py"),
            "--port",
            str(args.port),
            "policy:checkpoint",
            "--policy.config",
            f"pi05_robotwin_{args.task}",
            "--policy.dir",
            str(checkpoint),
        ]
        client_command = [
            os.path.abspath(args.client_python),
            "-u",
            str(robotwin / "pace/scripts/evaluate.py"),
            "--task",
            args.task,
            "--method",
            args.method,
            "--gpu",
            str(args.gpu),
            "--seed",
            str(args.seed),
            "--port",
            str(args.port),
            "--checkpoint-id",
            str(checkpoint),
            "--output",
            str(output / "evaluation"),
        ]
        if args.method == "fixed":
            client_command.extend(["--horizon", str(args.horizon)])
        if args.ultrafast:
            client_command.append("--ultrafast")
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_MEM_FRACTION="0.2"
        )
        original_path = env.get("PYTHONPATH", "")
        client_path = str(openpi / "packages/openpi-client/src")
        server_env = dict(
            env,
            PYTHONPATH=os.pathsep.join(
                [str(openpi / "src"), str(openpi / "pace/src"), client_path, original_path]
            ),
        )
        client_env = dict(env, PYTHONPATH=os.pathsep.join([client_path, original_path]))
        launch.update(server_command=server_command, client_command=client_command)
        write_json(output / "launch.json", launch)
        with (
            (output / "server.log").open("w") as server_log,
            (output / "client.log").open("w") as client_log,
        ):
            launch["server_started_utc"] = utc_now()
            write_json(output / "launch.json", launch)
            server = subprocess.Popen(
                server_command,
                cwd=openpi,
                env=server_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(server, args.port, args.startup_timeout)
            launch.update(status="evaluating", server_ready_utc=utc_now())
            write_json(output / "launch.json", launch)
            client = subprocess.Popen(
                client_command,
                cwd=robotwin,
                env=client_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            launch["client_started_utc"] = utc_now()
            write_json(output / "launch.json", launch)
            reader = threading.Thread(
                target=copy_progress, args=(client.stdout, client_log, label), daemon=True
            )
            reader.start()
            try:
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError("policy server exited during evaluation; see server.log")
                    time.sleep(0.2)
                if client.returncode != 0:
                    raise RuntimeError(
                        f"evaluation exited with code {client.returncode}; see client.log"
                    )
            finally:
                stop(client)
                reader.join(timeout=5)
                client.stdout.close()
                client = None
            result_path = output / "evaluation/metadata.json"
            result = json.loads(result_path.read_text())
            episodes = [
                json.loads(line)
                for line in (output / "evaluation/episodes.jsonl").read_text().splitlines()
            ]
            if (
                result["status"] != "complete"
                or result["episodes"] != 100
                or len(episodes) != 100
                or [e["episode"] for e in episodes] != list(range(100))
                or any(type(e["success"]) is not bool for e in episodes)
                or sum(e["success"] for e in episodes) != result["successes"]
            ):
                raise RuntimeError("evaluation did not produce 100 consistent episode records")
            launch.update(status="evaluated", episodes=100, successes=result["successes"])
    except BaseException as error:
        launch.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            try:
                stop(client)
            finally:
                stop(server)
        except BaseException as error:
            launch.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if launch["status"] == "evaluated":
                launch["status"] = "complete"
            try:
                result_path = output / "evaluation/metadata.json"
                if launch["status"] == "failed" and result_path.exists():
                    result = json.loads(result_path.read_text())
                    result.update(status="failed", error=launch["error"])
                    write_json(result_path, result)
            except BaseException as error:
                launch.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                launch["finished_utc"] = utc_now()
                write_json(output / "launch.json", launch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "server-python", "client-python"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("checkpoint-root", "openpi-root", "robotwin-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("gpu", "port", "seed"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--method", choices=("pace", "fixed"), required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--ultrafast", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as error:
        print(f"FAILED {args.task}: {error}; logs: {args.output}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
