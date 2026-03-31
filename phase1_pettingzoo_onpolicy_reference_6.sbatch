#!/bin/bash
#SBATCH --job-name=phase1_pz_onref6
#SBATCH -A ece567w26_class
#SBATCH --qos=class
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/home/naikewu/RL/BenchMARL/logs/%x-%A_%a.out
#SBATCH --error=/home/naikewu/RL/BenchMARL/logs/%x-%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

ROOT="/home/naikewu/RL"
REPO_DIR="${ROOT}/BenchMARL"
LOG_DIR="${REPO_DIR}/logs"
OUT_ROOT="${REPO_DIR}/phase1_outputs"
CONDA_SH="/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh"
ENV_NAME="benchmarl_pz_v100"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

ALGS=(mappo ippo)
SEEDS=(0 1 2)

IDX="${SLURM_ARRAY_TASK_ID}"

ALG_IDX=$(( IDX / ${#SEEDS[@]} ))
SEED_IDX=$(( IDX % ${#SEEDS[@]} ))

TASK_NAME="simple_reference"
ALG="${ALGS[$ALG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
TASK="pettingzoo/${TASK_NAME}"
RUN_NAME="${ALG}__${TASK_NAME}__seed${SEED}"
RUN_DIR="${OUT_ROOT}/${TASK_NAME}/${ALG}/seed_${SEED}"

mkdir -p "${RUN_DIR}"

echo "===================================================="
echo "Phase 1 PettingZoo on-policy reference supplement"
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
  experiment.evaluation_episodes=5 \
  experiment.evaluation_deterministic_actions=True \
  "experiment.loggers=[csv]" \
  experiment.checkpoint_interval=0 \
  experiment.create_json=True \
  experiment.save_folder="${RUN_DIR}" \
  "hydra.run.dir=${RUN_DIR}"
