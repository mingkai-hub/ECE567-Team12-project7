#!/usr/bin/env python3
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_A = Path("/home/mingkai/BenchMARL/outputs_stage_a")
ROOT_B = Path("/home/mingkai/BenchMARL/outputs_stage_b_round2")
LOG_ROOT = Path("/home/mingkai/BenchMARL/logs")
OUT = Path("/home/mingkai/BenchMARL/plots/report_2x2_stageA_stageB_train.png")
OUT_REPORT = Path("/home/mingkai/BenchMARL/plots/report_2x2_stageA_stageB_train_clean.png")

COLORS_ALG = {"ippo": "#1f77b4", "mappo": "#d62728"}
DISPLAY_ALG = {"ippo": "IPPO", "mappo": "MAPPO"}

COLORS_VAR = {
    "full": "#1f77b4",
    "no_vclip": "#2ca02c",
    "no_pclip": "#d62728",
    "no_clip": "#9467bd",
    "no_clip_low_lr": "#ff7f0e",
}
DISPLAY_VAR = {
    "full": "full",
    "no_vclip": "no-value-clipping",
    "no_pclip": "no-policy-clipping",
    "no_clip": "no-clipping",
    "no_clip_low_lr": "no-clip-low-lr",
}


def read_series(path: Path):
    rows = []
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    rows.append((int(float(row[0])), float(row[1])))
                except ValueError:
                    continue
    return rows


def aggregate(runs):
    values_by_x = defaultdict(list)
    for run in runs:
        for x, y in run:
            values_by_x[x].append(y)
    xs = sorted(values_by_x)
    means = [sum(values_by_x[x]) / len(values_by_x[x]) for x in xs]
    stds = []
    for x in xs:
        vals = values_by_x[x]
        if len(vals) > 1:
            mu = sum(vals) / len(vals)
            stds.append((sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5)
        else:
            stds.append(0.0)
    return xs, means, stds


def load_completed_stage_a_extra():
    completed = set()
    for out_path in sorted(LOG_ROOT.glob("stageA_extra-46250886_*.out")):
        out_text = out_path.read_text(errors="replace")
        err_text = out_path.with_suffix(".err").read_text(errors="replace")
        m_task = re.search(r"TASK=(.*)", out_text)
        m_alg = re.search(r"ALG=(.*)", out_text)
        m_seed = re.search(r"SEED=(.*)", out_text)
        if not (m_task and m_alg and m_seed):
            continue
        is_completed = "334/334" in err_text or "100%|██████████| 334/334" in err_text
        if is_completed:
            completed.add((m_task.group(1).strip(), m_alg.group(1).strip(), m_seed.group(1).strip()))
    return completed


def load_stage_a_runs(task: str, alg: str, completed_extra):
    runs = []
    task_path = ROOT_A / alg / task
    for seed_dir in sorted(task_path.glob("seed_*")):
        seed = seed_dir.name.replace("seed_", "")
        if int(seed) >= 3 and (task, alg, seed) not in completed_extra:
            continue
        matches = list(seed_dir.glob("*/*/scalars/collection_reward_episode_reward_mean.csv"))
        if not matches:
            continue
        metric_path = matches[0]
        scalars_dir = metric_path.parent
        frame_map = dict(read_series(scalars_dir / "counters_total_frames.csv"))
        metric_rows = read_series(metric_path)
        series = [(frame_map[idx], val) for idx, val in metric_rows if idx in frame_map]
        if series:
            runs.append(series)
    return runs


def load_round2_maps():
    completed = set()
    for pattern in ["stageB_nav_r2-46096255_*.out", "stageB_r2_extra-46251297_*.out"]:
        for out_path in sorted(LOG_ROOT.glob(pattern)):
            out_text = out_path.read_text(errors="replace")
            err_text = out_path.with_suffix(".err").read_text(errors="replace")
            m_variant = re.search(r"VARIANT=(.*)", out_text)
            m_seed = re.search(r"SEED=(.*)", out_text)
            if not (m_variant and m_seed):
                continue
            variant = m_variant.group(1).strip()
            seed = m_seed.group(1).strip()
            is_completed = "334/334" in err_text or "100%|██████████| 334/334" in err_text
            is_failed = any(x in err_text for x in ["Traceback", "AssertionError", "RuntimeError", "Exception", "action.isnan"])
            if is_completed and not is_failed:
                completed.add((variant, seed))
    return completed


def load_stage_b_runs(variant: str, completed_round2):
    runs = []
    for seed in ["0", "1", "2", "3", "4", "5"]:
        if (variant, seed) not in completed_round2:
            continue
        seed_root = ROOT_B / variant / "vmas" / "navigation" / f"seed_{seed}"
        matches = list(seed_root.glob("*/*/scalars/collection_reward_episode_reward_mean.csv"))
        if not matches:
            continue
        metric_path = matches[0]
        scalars_dir = metric_path.parent
        frame_map = dict(read_series(scalars_dir / "counters_total_frames.csv"))
        metric_rows = read_series(metric_path)
        series = [(frame_map[idx], val) for idx, val in metric_rows if idx in frame_map]
        if series:
            runs.append(series)
    return runs


def main():
    completed_extra = load_completed_stage_a_extra()
    completed_round2 = load_round2_maps()

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 6.3))
    ax_balance, ax_nav, ax_flocking, ax_b = axes.flatten()

    for ax, task, title in [
        (ax_balance, "vmas/balance", "VMAS balance"),
        (ax_nav, "vmas/navigation", "VMAS navigation"),
        (ax_flocking, "vmas/flocking", "VMAS flocking"),
    ]:
        for alg in ["ippo", "mappo"]:
            runs = load_stage_a_runs(task, alg, completed_extra)
            if not runs:
                continue
            xs, means, stds = aggregate(runs)
            ax.plot(xs, means, color=COLORS_ALG[alg], linewidth=2.2, label=f"{DISPLAY_ALG[alg]} (n={len(runs)})")
            ax.fill_between(xs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=COLORS_ALG[alg], alpha=0.16)
        ax.set_title(title)
        ax.set_xlabel("Training Frames")
        ax.set_ylabel("Training Reward")
        ax.grid(True, alpha=0.25)
        ax.legend(
            fontsize=12,
            frameon=False,
            borderpad=0.2,
            labelspacing=0.25,
            handlelength=1.8,
            handletextpad=0.5,
            loc="lower right",
        )

    for variant in ["full", "no_vclip", "no_pclip", "no_clip", "no_clip_low_lr"]:
        runs = load_stage_b_runs(variant, completed_round2)
        if not runs:
            continue
        xs, means, stds = aggregate(runs)
        ax_b.plot(xs, means, color=COLORS_VAR[variant], linewidth=2.2, label=f"{DISPLAY_VAR[variant]} (n={len(runs)})")
        ax_b.fill_between(xs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], color=COLORS_VAR[variant], alpha=0.14)
    ax_b.set_title("IPPO clipping ablations on VMAS navigation")
    ax_b.set_xlabel("Training Frames")
    ax_b.set_ylabel("Training Reward")
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(
        fontsize=12,
        frameon=False,
        borderpad=0.2,
        labelspacing=0.25,
        handlelength=1.8,
        handletextpad=0.5,
        loc="lower right",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_REPORT, dpi=220, bbox_inches="tight")
    print(OUT)
    print(OUT_REPORT)


if __name__ == "__main__":
    main()
