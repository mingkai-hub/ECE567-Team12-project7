# Project 7 Reproducibility Guide

This document describes the experiment scripts and plotting scripts used for our cooperative MARL reproduction study in BenchMARL on PettingZoo.

## Scope

We study the main claim of the MAPPO paper:

- Chao Yu et al., *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*  
  https://arxiv.org/abs/2103.01955

Our experiments use:

- Framework: BenchMARL
- Environment family: PettingZoo
- Tasks:
  - `simple_spread`
  - `simple_reference`

## Phase 1

Phase 1 compares four baselines:

- MAPPO
- IPPO
- VDN
- IQL

### Main scripts

- `phase1_pettingzoo_24.sbatch`
  Main Phase 1 comparison script.
- `phase1_pettingzoo_offpolicy_12.sbatch`
  Off-policy rerun script for VDN/QMIX-family settings.
- `phase1_pettingzoo_iql_6.sbatch`
  IQL supplement script.
- `phase1_pettingzoo_onpolicy_reference_6.sbatch`
  MAPPO/IPPO runs on `simple_reference`.
- `phase1_pettingzoo_extra_seeds_24.sbatch`
  Extra seeds for Phase 1.

### Plotting

- `plot_phase1_curves.py`
  General Phase 1 plotting utility.
- `plot_all_completed_three_panel.py`
  Three-panel evaluation figure.
- `plot_all_completed_three_panel_training.py`
  Three-panel training figure.

## Phase 2

Phase 2 performs MAPPO ablations on `pettingzoo/simple_spread`.

### Ablations

- PPO clip epsilon
- PPO update iterations
- Minibatch-size-related setting
- Batch size

### Main scripts

- `phase2_mappo_ablation_24.sbatch`
  Original Phase 2 ablation runs.
- `phase2_mappo_ablation_extra_seeds_16.sbatch`
  Additional seeds for Phase 2.

### Auxiliary plotting scripts

- `plot_phase2_clip_progress.py`
- `plot_phase2_progress_overview.py`
- `plot_phase2_completed_0_20.py`
- `plot_phase2_completed_all.py`
- `plot_phase2_completed_n_not_3.py`
- `plot_phase2_first_version_all.py`

## Notes

- Large outputs, logs, and intermediate result folders are intentionally excluded from version control.
- The files listed here are the scripts needed to reproduce the runs and figures used in our project workflow.
