# Multi-agent Reinforcement Learning

## Reproducing IPPO Phenomena in BenchMARL on VMAS

This repository studies whether the main empirical observations from the IPPO paper also appear in BenchMARL on VMAS tasks.

The original IPPO paper evaluates on SMAC. In this project, the goal is not to exactly reproduce the original numerical scores, but to reproduce the paper's qualitative findings in a different multi-agent benchmark.

## Attribution

This project is built on top of the original BenchMARL framework rather than being a standalone MARL library. The original BenchMARL project can be found here:

- GitHub: https://github.com/facebookresearch/BenchMARL
- Documentation: https://benchmarl.readthedocs.io/
- IPPO paper: https://doi.org/10.48550/arXiv.2011.09533

This repository contains our reproduction-oriented modifications, experiment scripts, and analysis code for the course project.

## Project Goal

We focus on three claims from the IPPO paper:

1. Independent-critic methods such as IPPO can be competitive with more complex centralized methods.
2. The benefit of a centralized critic is task-dependent rather than universal.
3. Policy clipping is an important source of training stability, and reducing the learning rate alone is not a full replacement.

## Environment and Baselines

We use VMAS tasks through BenchMARL, with a focus on cooperative continuous-control settings. The main Stage A tasks are:

- `vmas/balance`
- `vmas/navigation`
- `vmas/dispersion`
- `vmas/flocking`

For the main comparison, we use:

- `IPPO` as the independent-critic baseline
- `MAPPO` as the centralized-critic baseline

For the clipping study, we use IPPO ablations on `vmas/navigation`:

- `full`
- `no-value-clipping`
- `no-policy-clipping`
- `no-clipping`
- `no-clip-low-lr`

## Experimental Setup

### Stage A

To compare IPPO and MAPPO fairly, we keep the main experimental setup fixed across both methods. We use the same default VMAS task configuration, the same MLP model for actor and critic, a total training budget of 2 million frames, evaluation every 120k frames, and 5 evaluation episodes. We also keep the random seeds, CSV logging setup, and compute-device settings the same.

### Stage B

For the clipping ablation study, we use the same `vmas/navigation` task and the same total budget of 2 million frames, while varying policy clipping and value clipping. In the more conservative second round, we use 10 evaluation episodes, a smaller learning rate, and fewer minibatch iterations to improve stability.

## Main Findings

### Stage A

The current Stage A results support the claim that the usefulness of a centralized critic is task-dependent.

- On `VMAS balance`, MAPPO is clearly stronger and more stable than IPPO.
- On `VMAS navigation`, IPPO is competitive with MAPPO and appears slightly stronger in the completed runs.
- On `VMAS dispersion`, the two methods are very similar.
- On `VMAS flocking`, IPPO outperforms MAPPO by a noticeable margin.

Taken together, these results do not support the idea that MAPPO is uniformly better than IPPO. Instead, they are more consistent with the IPPO paper's observation that independent-critic methods can remain competitive, and that centralized critics help in some tasks but not all tasks.

### Stage B

The Stage B ablation results support the clipping-related conclusion of the IPPO paper.

- In the first ablation round, removing policy clipping led to repeated NaN-action failures.
- In a more conservative second round, the previously unstable variants became much more trainable.
- However, even after stabilization, the unclipped variants still underperformed the clipped baselines.

This suggests that policy clipping is a major contributor to stable training, and that lowering the learning rate alone does not fully replace clipping.

## Final Figures

Evaluation figure:

- [report_2x2_stageA_stageB_clean.png](plots/report_2x2_stageA_stageB_clean.png)

Training figure:

- [report_2x2_stageA_stageB_train_clean.png](plots/report_2x2_stageA_stageB_train_clean.png)

## Repository Contents

Key files in this repository:

- `benchmarl/algorithms/ippo.py`
  Modified IPPO implementation with separate policy-clipping and value-clipping controls.
- `benchmarl/conf/algorithm/ippo.yaml`
  IPPO configuration used for the clipping study.
- `plot_report_2x2.py`
  Generates the final 2x2 evaluation figure used in the report.
- `plot_report_2x2_train.py`
  Generates the final 2x2 training figure used in the report.
- `task_list.txt`
  Project task plan and experiment organization notes.

## Reproducing Effectiveness of PPO in Cooperative, Multi-Agent Games

This section documents our later Project 7 reproduction work on cooperative PettingZoo tasks in BenchMARL. It is added without modifying the original project description above.

### Reference Paper

- Chao Yu et al., *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*
- arXiv: https://arxiv.org/abs/2103.01955

### Framework and Tasks

We use BenchMARL with PettingZoo and focus on:

- `simple_spread`
- `simple_reference`

### Phase 1

Phase 1 compares four baselines:

- MAPPO
- IPPO
- VDN
- IQL

The main experiment configuration uses:

- discrete actions
- 2M total environment frames
- evaluation every 120k frames
- 5 evaluation episodes
- MLP policy/critic with hidden sizes `[256, 256]`

### Phase 2

Phase 2 studies MAPPO ablations on `pettingzoo/simple_spread`, focusing on:

- PPO clip epsilon
- update iterations
- minibatch-related settings
- batch size

### Reproduction Scripts Added in This Repository

Plotting scripts:

- `plot_phase1_curves.py`
- `plot_phase2_clip_progress.py`
- `plot_phase2_progress_overview.py`
- `plot_phase2_completed_0_20.py`
- `plot_phase2_completed_all.py`
- `plot_phase2_completed_n_not_3.py`
- `plot_phase2_first_version_all.py`
- `plot_all_completed_three_panel.py`
- `plot_all_completed_three_panel_training.py`

Experiment documentation:

- `experiment_matrix.txt`
- `PROJECT7_PETTINGZOO_REPRO.md`

