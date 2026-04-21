#!/usr/bin/env bash

# Do not exit the whole script when a child job returns non-zero.
# Keep -u and pipefail to catch unset variables and pipeline errors.
set -uo pipefail

# Base logs directory (per-dataset override applied inside loop)
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

run_job_split_step() {
	local gpu="$1"
	local script="$2"
	local split="$3"
	local step="$4"

	local base
	base="$(basename "$script" .py)"
	local log_file="$LOG_DIR/${base}_${split}.log"

	echo "[$(date '+%F %T')] GPU=${gpu} start: ${script} --split ${split} --step ${step}"
	CUDA_VISIBLE_DEVICES="$gpu" nohup python "$script" --split "$split" --step "$step" > "$log_file" 2>&1 &
	local pid=$!
	echo "[$(date '+%F %T')] PID=${pid} log=${log_file}"

	# Keep one running job per GPU queue; start next only after current finishes.
	wait "$pid"
	local rc=$?
	echo "[$(date '+%F %T')] GPU=${gpu} done: ${script} --split ${split} (exit=${rc})"
	return "$rc"
}

run_queue_split_step() {
	local gpu="$1"
	local script="$2"
	local split_a="$3"
	local split_b="$4"
	local step="$5"

	run_job_split_step "$gpu" "$script" "$split_a" "$step"
	run_job_split_step "$gpu" "$script" "$split_b" "$step"
}

run_job_num_splits() {
	local gpu="$1"
	local script="$2"
	local num_of_splits="$3"

	local base
	base="$(basename "$script" .py)"
	local log_file="$LOG_DIR/${base}.log"

	echo "[$(date '+%F %T')] GPU=${gpu} start: ${script} --num_of_splits ${num_of_splits}"
	CUDA_VISIBLE_DEVICES="$gpu" nohup python "$script" --num_of_splits "$num_of_splits" > "$log_file" 2>&1 &
	local pid=$!
	echo "[$(date '+%F %T')] PID=${pid} log=${log_file}"

	wait "$pid"
	local rc=$?
	echo "[$(date '+%F %T')] GPU=${gpu} done: ${script} --num_of_splits ${num_of_splits} (exit=${rc})"
	return "$rc"
}

STEP=1250

# Number of GPUs available. Set via env var NUM_GPUS=4 for 4-GPU mode.
NUM_GPUS=${NUM_GPUS:-2}

usage() {
	cat <<'USAGE'
Usage: ./run.sh [--dataset NAME] [--datasets NAME1,NAME2,...] [--step N]

Options:
  --dataset, -d   Single dataset name (default: coco). Can be specified multiple times.
  --datasets      Comma-separated dataset names (e.g., pick-a-pic,diffusiondb).
	--step           Denoising step for *small_steps scripts (default: 1250).
USAGE
}

DATASETS=()
while [[ $# -gt 0 ]]; do
	case "$1" in
		--step)
			STEP="$2"
			shift 2
			;;
		--dataset|-d)
			DATASETS+=("$2")
			shift 2
			;;
		--datasets)
			IFS=',' read -r -a _tmp_datasets <<< "$2"
			for _ds in "${_tmp_datasets[@]}"; do
				DATASETS+=("$_ds")
			done
			shift 2
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

if [ "${#DATASETS[@]}" -eq 0 ]; then
	DATASETS=("coco")
fi

update_dataset_name() {
	local ds="$1"
	python - <<PY
import re

path = "config.py"
ds = ${ds@Q}

with open(path, "r", encoding="utf-8") as f:
	data = f.read()

new_data, count = re.subn(r"^(\s*DATASET_NAME\s*=\s*).*$", r"\\1" + repr(ds), data, flags=re.M)
if count == 0:
	raise SystemExit("DATASET_NAME not found in config.py")

with open(path, "w", encoding="utf-8") as f:
	f.write(new_data)
PY
}

# helper to launch jobs without waiting and collect pids
JOB_PIDS=()
launch_job_split_step_no_wait() {
	local gpu="$1"
	local script="$2"
	local split="$3"
	local step="$4"
	local base
	base="$(basename "$script" .py)"
	local log_file="$LOG_DIR/${base}_${split}.log"
	echo "[$(date '+%F %T')] GPU=${gpu} launch: ${script} --split ${split} --step ${step}"
	CUDA_VISIBLE_DEVICES="$gpu" nohup python "$script" --split "$split" --step "$step" > "$log_file" 2>&1 &
	JOB_PIDS+=("$!")
}

launch_job_num_splits_no_wait() {
	local gpu="$1"
	local script="$2"
	local num_of_splits="$3"
	local base
	base="$(basename "$script" .py)"
	local log_file="$LOG_DIR/${base}.log"
	echo "[$(date '+%F %T')] GPU=${gpu} launch: ${script} --num_of_splits ${num_of_splits}"
	CUDA_VISIBLE_DEVICES="$gpu" nohup python "$script" --num_of_splits "$num_of_splits" > "$log_file" 2>&1 &
	JOB_PIDS+=("$!")
}

wait_all_jobs() {
	for pid in "${JOB_PIDS[@]:-}"; do
		wait "$pid"
	done
	JOB_PIDS=()
}

run_pipeline() {
	echo "===== Stage 1: traj_all (per-GPU queues) ====="

	if [ "$NUM_GPUS" -eq 2 ]; then
		# GPU0: traj_all split0 -> split1
		(
			run_job_split_step 0 ./traj_all.py 0 "$STEP"
			run_job_split_step 0 ./traj_all.py 1 "$STEP"
		) &
		pid_traj_gpu0=$!

		# GPU1: traj_all split2 -> split3
		(
			run_job_split_step 1 ./traj_all.py 2 "$STEP"
			run_job_split_step 1 ./traj_all.py 3 "$STEP"
		) &
		pid_traj_gpu1=$!

		wait "$pid_traj_gpu0"
		wait "$pid_traj_gpu1"
	else
		# NUM_GPUS==4: launch 4 traj_all in parallel on GPUs 0..3
		JOB_PIDS=()
		launch_job_split_step_no_wait 0 ./traj_all.py 0 "$STEP"
		launch_job_split_step_no_wait 1 ./traj_all.py 1 "$STEP"
		launch_job_split_step_no_wait 2 ./traj_all.py 2 "$STEP"
		launch_job_split_step_no_wait 3 ./traj_all.py 3 "$STEP"
		wait_all_jobs
	fi

	echo "===== Stage 2: retrieve (run after all traj_all complete) ====="
	if [ "$NUM_GPUS" -eq 2 ]; then
		# run retrieve on GPU0 and retrieve_by_prompt on GPU1 in parallel, wait both
		CUDA_VISIBLE_DEVICES=0 nohup python ./retrieve.py --num_of_splits 4 > "$LOG_DIR/retrieve.log" 2>&1 &
		pid_retrieve0=$!
		CUDA_VISIBLE_DEVICES=1 nohup python ./retrieve_by_prompt.py --num_of_splits 4 > "$LOG_DIR/retrieve_by_prompt.log" 2>&1 &
		pid_retrieve1=$!
		wait "$pid_retrieve0"
		wait "$pid_retrieve1"
	else
		# NUM_GPUS==4: still run the two retrieves in parallel (use GPU0 and GPU1)
		JOB_PIDS=()
		launch_job_num_splits_no_wait 0 ./retrieve.py 4
		launch_job_num_splits_no_wait 1 ./retrieve_by_prompt.py 4
		wait_all_jobs
	fi


	echo "===== Stage 3: traj_small_steps.py ====="
	if [ "$NUM_GPUS" -eq 2 ]; then
		run_queue_split_step 0 ./traj_small_steps.py 0 2 "$STEP" &
		pid_q0=$!
		run_queue_split_step 1 ./traj_small_steps.py 1 3 "$STEP" &
		pid_q1=$!
		wait "$pid_q0"
		wait "$pid_q1"
	else
		# NUM_GPUS==4: launch four splits in parallel on GPUs 0..3
		JOB_PIDS=()
		launch_job_split_step_no_wait 0 ./traj_small_steps.py 0 "$STEP"
		launch_job_split_step_no_wait 1 ./traj_small_steps.py 1 "$STEP"
		launch_job_split_step_no_wait 2 ./traj_small_steps.py 2 "$STEP"
		launch_job_split_step_no_wait 3 ./traj_small_steps.py 3 "$STEP"
		wait_all_jobs
	fi

	echo "===== Stage 4: attention_small_steps.py ====="
	if [ "$NUM_GPUS" -eq 2 ]; then
		run_queue_split_step 0 ./attention_small_steps.py 0 2 "$STEP" &
		pid_q0=$!
		run_queue_split_step 1 ./attention_small_steps.py 1 3 "$STEP" &
		pid_q1=$!
		wait "$pid_q0"
		wait "$pid_q1"
	else
		JOB_PIDS=()
		launch_job_split_step_no_wait 0 ./attention_small_steps.py 0 "$STEP"
		launch_job_split_step_no_wait 1 ./attention_small_steps.py 1 "$STEP"
		launch_job_split_step_no_wait 2 ./attention_small_steps.py 2 "$STEP"
		launch_job_split_step_no_wait 3 ./attention_small_steps.py 3 "$STEP"
		wait_all_jobs
	fi

	echo "===== Stage 5: nirvana/recon/redi/modm/ours/ours_mix ====="
	if [ "$NUM_GPUS" -eq 2 ]; then
		# GPU0 队列（串行：nirvana -> redi -> ours）
		(
			run_job_num_splits 0 ./nirvana_generate.py 4
			run_job_num_splits 0 ./redi_generate.py 4
			run_job_num_splits 0 ./ours_generate_tt.py 4
			run_job_num_splits 0 ./ours_generate_mix_tt.py 4
		) &
		pid_gpu0=$!

		# GPU1 队列（串行：recon -> modm -> ours_mix）
		(
			run_job_num_splits 1 ./recon_generate.py 4
			run_job_num_splits 1 ./modm_generate.py 4
			run_job_num_splits 1 ./ours_generate_mix_ti.py 4
			run_job_num_splits 1 ./ours_generate_ti.py 4
		) &
		pid_gpu1=$!

		wait "$pid_gpu0"
		wait "$pid_gpu1"
	else
		# NUM_GPUS==4: run in two waves (one job per GPU)
		JOB_PIDS=()
		wave1=("./nirvana_generate.py" "./recon_generate.py" "./redi_generate.py" "./modm_generate.py")
		for i in "${!wave1[@]}"; do
			script="${wave1[$i]}"
			gpu=$i
			launch_job_num_splits_no_wait "$gpu" "$script" 4
			done
		wait_all_jobs

		JOB_PIDS=()
		wave2=("./ours_generate_tt.py" "./ours_generate_mix_tt.py" "./ours_generate_ti.py" "./ours_generate_mix_ti.py")
		for i in "${!wave2[@]}"; do
			script="${wave2[$i]}"
			gpu=$i
			launch_job_num_splits_no_wait "$gpu" "$script" 4
			done
		wait_all_jobs
	fi

	echo "===== Stage 6: compare.py ====="
	CUDA_VISIBLE_DEVICES=0 nohup python ./compare.py > "$LOG_DIR/compare.log" 2>&1 &
	pid_cmp=$!
	wait "$pid_cmp"

	echo "All queued jobs finished."
}

for ds in "${DATASETS[@]}"; do
	echo "===== Dataset: ${ds} ====="
	update_dataset_name "$ds"

	# Use a per-dataset log directory so logs are grouped by dataset name
	LOG_DIR="./logs/${ds}"
	mkdir -p "$LOG_DIR"

	run_pipeline
done
