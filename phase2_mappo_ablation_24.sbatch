#!/bin/bash
#SBATCH --job-name=phase2_mappo_ab24
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
OUT_ROOT="${REPO_DIR}/phase2_mappo_ablation_outputs"
CONDA_SH="/sw/pkgs/arc/python3.11-anaconda/2024.02-1/etc/profile.d/conda.sh"
ENV_NAME="benchmarl_pz_v100"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

source "${CONDA_SH}"
conda activate "${ENV_NAME}"

TASK="pettingzoo/simple_spread"
ALG="mappo"
SEEDS=(0 1 2)

GROUP_NAMES=(clip_epsilon on_policy_n_minibatch_iters on_policy_minibatch_size on_policy_collected_frames_per_batch)
VALUE_A=(0.05 15 200 3000)
VALUE_B=(0.1 90 800 12000)
VALUE_A_LABELS=(0p05 15 200 3000)
VALUE_B_LABELS=(0p1 90 800 12000)

IDX="${SLURM_ARRAY_TASK_ID}"

GROUP_IDX=$(( IDX / ( 2 * ${#SEEDS[@]} ) ))
REM=$(( IDX % ( 2 * ${#SEEDS[@]} ) ))
VALUE_IDX=$(( REM / ${#SEEDS[@]} ))
SEED_IDX=$(( REM % ${#SEEDS[@]} ))

GROUP_NAME="${GROUP_NAMES[$GROUP_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

if [ "${VALUE_IDX}" -eq 0 ]; then
  VALUE="${VALUE_A[$GROUP_IDX]}"
  VALUE_LABEL="${VALUE_A_LABELS[$GROUP_IDX]}"
else
  VALUE="${VALUE_B[$GROUP_IDX]}"
  VALUE_LABEL="${VALUE_B_LABELS[$GROUP_IDX]}"
fi

OVERRIDE_KEY=""
OVERRIDE_ARG=""
case "${GROUP_NAME}" in
  clip_epsilon)
    OVERRIDE_KEY="algorithm.clip_epsilon"
    OVERRIDE_ARG="algorithm.clip_epsilon=${VALUE}"
    ;;
  on_policy_n_minibatch_iters)
    OVERRIDE_KEY="experiment.on_policy_n_minibatch_iters"
    OVERRIDE_ARG="experiment.on_policy_n_minibatch_iters=${VALUE}"
    ;;
  on_policy_minibatch_size)
    OVERRIDE_KEY="experiment.on_policy_minibatch_size"
    OVERRIDE_ARG="experiment.on_policy_minibatch_size=${VALUE}"
    ;;
  on_policy_collected_frames_per_batch)
    OVERRIDE_KEY="experiment.on_policy_collected_frames_per_batch"
    OVERRIDE_ARG="experiment.on_policy_collected_frames_per_batch=${VALUE}"
    ;;
  *)
    echo "Unknown ablation group: ${GROUP_NAME}" >&2
    exit 1
    ;;
esac

RUN_NAME="${ALG}__simple_spread__${GROUP_NAME}_${VALUE_LABEL}__seed${SEED}"
RUN_DIR="${OUT_ROOT}/${GROUP_NAME}/${VALUE_LABEL}/seed_${SEED}"

mkdir -p "${RUN_DIR}"

echo "===================================================="
echo "Phase 2 MAPPO ablation"
echo "Array IDX=${IDX}"
echo "TASK=${TASK}"
echo "ALG=${ALG}"
echo "GROUP=${GROUP_NAME}"
echo "OVERRIDE_KEY=${OVERRIDE_KEY}"
echo "VALUE=${VALUE}"
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
  ${OVERRIDE_ARG} \
  experiment.save_folder="${RUN_DIR}" \
  "hydra.run.dir=${RUN_DIR}"
