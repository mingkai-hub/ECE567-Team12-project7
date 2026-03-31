#!/bin/bash
#SBATCH --job-name=phase1_pz_s345
#SBATCH -A ece567w26_class
#SBATCH --qos=class
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-23%1
#SBATCH --output=/home/naikewu/RL/BenchMARL/logs/%x-%A_%a.out
#SBATCH --error=/home/naikewu/RL/BenchMARL/logs/%x-%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

ROOT="/home/naikewu/RL"
REPO_DIR="${ROOT}/BenchMARL"
LOG_DIR="${REPO_DIR}/logs"
CONDA_SH="/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh"
ENV_NAME="benchmarl_pz_v100"

mkdir -p "${LOG_DIR}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

TASKS=(simple_spread simple_reference)
ALGS=(mappo ippo vdn iql)
SEEDS=(3 4 5)

IDX="${SLURM_ARRAY_TASK_ID}"

TASK_IDX=$(( IDX / ( ${#ALGS[@]} * ${#SEEDS[@]} ) ))
REM=$(( IDX % ( ${#ALGS[@]} * ${#SEEDS[@]} ) ))
ALG_IDX=$(( REM / ${#SEEDS[@]} ))
SEED_IDX=$(( REM % ${#SEEDS[@]} ))

TASK_NAME="${TASKS[$TASK_IDX]}"
ALG="${ALGS[$ALG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TASK="pettingzoo/${TASK_NAME}"

OUT_ROOT=""
EXTRA_ARGS=()
case "${ALG}" in
  mappo|ippo)
    OUT_ROOT="${REPO_DIR}/phase1_outputs"
    ;;
  vdn)
    OUT_ROOT="${REPO_DIR}/phase1_offpolicy_outputs"
    EXTRA_ARGS+=(
      experiment.off_policy_init_random_frames=20000
      experiment.off_policy_n_optimizer_steps=200
      experiment.off_policy_train_batch_size=256
    )
    ;;
  iql)
    OUT_ROOT="${REPO_DIR}/phase1_iql_outputs"
    EXTRA_ARGS+=(
      experiment.off_policy_init_random_frames=20000
      experiment.off_policy_n_optimizer_steps=200
      experiment.off_policy_train_batch_size=256
    )
    ;;
  *)
    echo "Unknown algorithm: ${ALG}" >&2
    exit 1
    ;;
esac

RUN_NAME="${ALG}__${TASK_NAME}__seed${SEED}"
RUN_DIR="${OUT_ROOT}/${TASK_NAME}/${ALG}/seed_${SEED}"

mkdir -p "${RUN_DIR}"

echo "===================================================="
echo "Phase 1 PettingZoo extra seeds"
echo "Array IDX=${IDX}"
echo "TASK=${TASK}"
echo "ALG=${ALG}"
echo "SEED=${SEED}"
echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
date
nvidia-smi || true
echo "===================================================="

cd "${REPO_DIR}"

python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"

python -m benchmarl.run \
  algorithm="${ALG}" task="${TASK}" seed="${SEED}" \
  model=layers/mlp model@critic_model=layers/mlp \
  "model.num_cells=[256,256]" \
  "critic_model.num_cells=[256,256]" \
  model.activation_class=torch.nn.Tanh \
  critic_model.activation_class=torch.nn.Tanh \
  experiment.prefer_continuous_actions=False \
  experiment.share_policy_params=True \
  experiment.sampling_device=cpu \
  experiment.train_device=cuda \
  experiment.buffer_device=cpu \
  experiment.max_n_frames=2000000 \
  experiment.evaluation=True \
  experiment.render=False \
  experiment.evaluation_interval=120000 \
  experiment.evaluation_episodes=10 \
  experiment.evaluation_deterministic_actions=True \
  "experiment.loggers=[csv]" \
  experiment.checkpoint_interval=0 \
  experiment.create_json=True \
  "${EXTRA_ARGS[@]}" \
  experiment.save_folder="${RUN_DIR}" \
  "hydra.run.dir=${RUN_DIR}"
