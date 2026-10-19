#!/usr/bin/env bash
# One method and evaluation seed, using the original task order and GPU slots.
set -euo pipefail

ROBOTWIN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fail() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash pace/scripts/eval_all.sh [options]

Evaluate each task once for 100 episodes with one seed.
Results: pace/runs/<start-time>/<task>/<method>/seed<N>/

  --method METHOD       pace (default) or fixed
  --seed N              0 (default), 1 or 2
  --horizon N           Action horizon for fixed (1-50)
  --gpus "0 1 ..."       GPU indices (default: "0 1 2 3 4 5 6 7")
  --jobs-per-gpu N       Concurrent evaluations per GPU (default: 3)
  --tasks "TASK ..."     Evaluate selected tasks in the original queue order
  --start-port N        First model-server port (default: 8000)
  --startup-timeout N   Model-server startup timeout in seconds (default: 600)
  --ultrafast           Use the released expert-screened scene cache
  --dry-run             Print the queue without starting evaluation
  -h, --help            Show this help

Edit machine paths in the adjacent eval.yaml. Activate the RoboTwin environment
before evaluation.
EOF
}

# Evaluation settings.
GPU_IDS="0 1 2 3 4 5 6 7"
MAX_PARALLEL_PER_GPU=3
METHOD=pace
HORIZON=""
SEED=0
TASKS=""
START_PORT=8000
STARTUP_TIMEOUT=600
SERVER_LAUNCH_SLEEP=2
POLL_SECONDS=5

dry_run=false
ultrafast=false
while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift; continue ;;
    --ultrafast) ultrafast=true; shift; continue ;;
    --method|--seed|--horizon|--gpus|--jobs-per-gpu|--tasks|--start-port|--startup-timeout)
      [[ $# -ge 2 && -n $2 && $2 != --* ]] || fail "Missing value for $1" ;;
    *) fail "Unknown option: $1. See --help." ;;
  esac
  case "$1" in
    --method) METHOD=$2 ;;
    --seed) SEED=$2 ;;
    --horizon) HORIZON=$2 ;;
    --gpus) GPU_IDS=$2 ;;
    --jobs-per-gpu) MAX_PARALLEL_PER_GPU=$2 ;;
    --tasks) TASKS=$2 ;;
    --start-port) START_PORT=$2 ;;
    --startup-timeout) STARTUP_TIMEOUT=$2 ;;
  esac
  shift 2
done
for value in "$MAX_PARALLEL_PER_GPU" "$START_PORT" "$STARTUP_TIMEOUT"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || fail "Concurrency, port and timeout must be positive integers"
done
[[ $SEED =~ ^[012]$ ]] || fail "--seed must be 0, 1 or 2"
case "$METHOD" in
  pace) [[ -z $HORIZON ]] || fail "PACE uses paper settings; omit --horizon"; method_name=pace ;;
  fixed)
    [[ $HORIZON =~ ^[1-9][0-9]*$ ]] && (( HORIZON <= 50 )) || fail "Fixed requires --horizon between 1 and 50"
    method_name="fixed_h$HORIZON" ;;
  *) fail "--method must be pace or fixed" ;;
esac
read -r -a gpus <<< "$GPU_IDS"
[[ ${#gpus[@]} -gt 0 ]] || fail "--gpus must contain GPU indices"
seen=" "
for gpu in "${gpus[@]}"; do
  [[ $gpu =~ ^(0|[1-9][0-9]*)$ && $seen != *" $gpu "* ]] || fail "Invalid or repeated GPU: $gpu"
  seen+="$gpu "
done
slots=$(( ${#gpus[@]} * MAX_PARALLEL_PER_GPU ))
(( START_PORT + slots - 1 <= 65535 )) || fail "Port range exceeds 65535"

# Use one local startup timestamp for the entire batch.
CLIENT_PYTHON=$(command -v python || true)
[[ -x $CLIENT_PYTHON ]] || fail "Activate your RoboTwin environment before running this script"
started_at=$("$CLIENT_PYTHON" -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d-%H%M%S-%f"))')
OUTPUT_ROOT="$ROBOTWIN_ROOT/pace/runs/$started_at"
config_paths=$("$CLIENT_PYTHON" "$ROBOTWIN_ROOT/pace/scripts/_eval_config.py" "$ROBOTWIN_ROOT") || exit $?
paths=()
while IFS= read -r path; do paths+=("$path"); done <<< "$config_paths"
OPENPI_ROOT=${paths[0]}
CHECKPOINT_ROOT=${paths[1]}
SERVER_PYTHON=${paths[2]}

task_order=()
while IFS= read -r task; do
  [[ -z $task ]] || task_order+=("$task")
done < "$ROBOTWIN_ROOT/pace/configs/task_order.txt"
if [[ -n $TASKS ]]; then
  read -r -a requested <<< "$TASKS"
  seen=" "
  for task in "${requested[@]}"; do
    found=false
    for known in "${task_order[@]}"; do [[ $task != "$known" ]] || found=true; done
    $found && [[ $seen != *" $task "* ]] || fail "Unknown or repeated task: $task"
    seen+="$task "
  done
  selected=()
  for task in "${task_order[@]}"; do
    [[ $seen != *" $task "* ]] || selected+=("$task")
  done
  task_order=("${selected[@]}")
fi

queue_output=()
for task in "${task_order[@]}"; do
  queue_output+=("$OUTPUT_ROOT/$task/$method_name/seed$SEED")
done
echo "Method=$method_name seed=$SEED | ${#task_order[@]} tasks x 100 episodes"
echo "GPUs: $GPU_IDS | $MAX_PARALLEL_PER_GPU per GPU | $slots slots | ports $START_PORT-$((START_PORT + slots - 1))"
echo "Results: $OUTPUT_ROOT"
echo "Task order: ${task_order[*]}"
echo "Service launch interval: ${SERVER_LAUNCH_SLEEP}s | Scheduler poll interval: ${POLL_SECONDS}s"
if $ultrafast; then
  echo "Expert screening: cached scenes from pace/configs/expert_scenes.json"
else
  echo "Expert screening: online"
fi
if $dry_run; then
  for (( i=0; i<${#task_order[@]}; i++ )); do
    echo "${task_order[$i]} | 100 episodes | ${queue_output[$i]}"
  done
  exit 0
fi

[[ -x $SERVER_PYTHON ]] || fail "Set server_python in pace/scripts/eval.yaml to your OpenPI Python interpreter"
[[ -f $OPENPI_ROOT/scripts/serve_policy.py ]] || fail "OpenPI official server script not found"
for task in "${task_order[@]}"; do
  [[ -d $CHECKPOINT_ROOT/$task/params && -f $CHECKPOINT_ROOT/$task/assets/robotwin/norm_stats.json ]] || fail "Missing checkpoint or normalization for $task"
done

slot_pid=(); slot_task=(); slot_output=()
for (( slot=0; slot<slots; slot++ )); do slot_pid+=(""); done
cleanup() {
  trap '' INT TERM
  for pid in "${slot_pid[@]}"; do
    [[ -z $pid ]] || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${slot_pid[@]}"; do
    [[ -z $pid ]] || wait "$pid" || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
next=0; finished=0; failures=0
while (( finished < ${#task_order[@]} )); do
  # Match the original scheduler: fill free slots from one ordered queue,
  # then inspect running jobs.
  for (( slot=0; slot<slots; slot++ )); do
    if [[ -z ${slot_pid[$slot]} ]] && (( next < ${#task_order[@]} )); then
      gpu=${gpus[$((slot / MAX_PARALLEL_PER_GPU))]}
      port=$((START_PORT + slot))
      task=${task_order[$next]}; output=${queue_output[$next]}
      command=("$CLIENT_PYTHON" -u "$ROBOTWIN_ROOT/pace/scripts/_eval_worker.py"
        --task "$task" --method "$METHOD" --seed "$SEED"
        --gpu "$gpu" --port "$port" --output "$output"
        --openpi-root "$OPENPI_ROOT" --robotwin-root "$ROBOTWIN_ROOT"
        --server-python "$SERVER_PYTHON" --client-python "$CLIENT_PYTHON"
        --checkpoint-root "$CHECKPOINT_ROOT" --startup-timeout "$STARTUP_TIMEOUT")
      if [[ $METHOD == fixed ]]; then command+=(--horizon "$HORIZON"); fi
      if $ultrafast; then command+=(--ultrafast); fi
      "${command[@]}" &
      slot_pid[$slot]=$!; slot_task[$slot]=$task; slot_output[$slot]=$output
      echo "[$(date '+%F %T %z')] START $task | $((next + 1))/${#task_order[@]} | GPU $gpu port $port slot $slot | PID ${slot_pid[$slot]} | $output"
      next=$((next + 1))
      if (( SERVER_LAUNCH_SLEEP > 0 )); then sleep "$SERVER_LAUNCH_SLEEP"; fi
    fi
  done
  for (( slot=0; slot<slots; slot++ )); do
    pid=${slot_pid[$slot]}
    if [[ -n $pid ]] && ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        echo "[$(date '+%F %T %z')] COMPLETE ${slot_task[$slot]} | $((finished + 1))/${#task_order[@]} | ${slot_output[$slot]}"
      else
        echo "[$(date '+%F %T %z')] FAILED ${slot_task[$slot]} | ${slot_output[$slot]}" >&2
        failures=$((failures + 1))
      fi
      slot_pid[$slot]=""
      finished=$((finished + 1))
    fi
  done
  if (( finished < ${#task_order[@]} )); then sleep "$POLL_SECONDS"; fi
done
echo "Finished $finished evaluations; technical failures: $failures"
(( failures == 0 ))
