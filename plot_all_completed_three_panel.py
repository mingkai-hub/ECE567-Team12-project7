#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


COMPLETE_FRAMES = 2_000_000


def read_scalar_csv(path: Path) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            rows.append((int(float(row[0])), float(row[1])))
    return rows


def pick_scalars_dir(seed_dir: Path) -> Path | None:
    candidates = sorted(seed_dir.rglob("scalars"))
    return candidates[-1] if candidates else None


def load_completed_curve_runs(seed_root: Path, curve_file: str) -> List[dict]:
    runs: List[dict] = []
    for seed_dir in sorted(p for p in seed_root.glob("seed_*") if p.is_dir()):
        scalars_dir = pick_scalars_dir(seed_dir)
        if scalars_dir is None:
            continue

        counters_path = scalars_dir / "counters_total_frames.csv"
        curve_path = scalars_dir / curve_file
        if not (counters_path.exists() and curve_path.exists()):
            continue

        counters = read_scalar_csv(counters_path)
        if not counters:
            continue
        last_frames = int(counters[-1][1])
        if last_frames < COMPLETE_FRAMES:
            continue

        frame_map = {int(idx): int(val) for idx, val in counters}
        curve = []
        for idx, value in read_scalar_csv(curve_path):
            if idx in frame_map:
                curve.append((frame_map[idx], value))
        if not curve:
            continue

        runs.append({"seed": seed_dir.name.replace("seed_", ""), "curve": curve})
    return runs


def aggregate_stats(curves: Iterable[List[Tuple[int, float]]]) -> List[Tuple[int, float, float]]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for curve in curves:
        for frame, value in curve:
            buckets[frame].append(value)
    stats: List[Tuple[int, float, float]] = []
    for frame, vals in sorted(buckets.items()):
        arr = np.asarray(vals, dtype=float)
        stats.append((frame, float(arr.mean()), float(arr.std(ddof=0))))
    return stats


def plot_panel(ax, series: Dict[str, dict], title: str) -> List[str]:
    summary = []
    for _, spec in series.items():
        runs = load_completed_curve_runs(spec["path"], "eval_reward_episode_reward_mean.csv")
        if not runs:
            continue

        for run in runs:
            xs = [x for x, _ in run["curve"]]
            ys = [y for _, y in run["curve"]]
            ax.plot(xs, ys, color=spec["color"], alpha=0.18, linewidth=0.9)

        stats_curve = aggregate_stats(run["curve"] for run in runs)
        xs = [x for x, _, _ in stats_curve]
        mean_ys = [y for _, y, _ in stats_curve]
        std_ys = [s for _, _, s in stats_curve]
        lower = [m - s for m, s in zip(mean_ys, std_ys)]
        upper = [m + s for m, s in zip(mean_ys, std_ys)]

        ax.fill_between(xs, lower, upper, color=spec["color"], alpha=0.10)
        ax.plot(
            xs,
            mean_ys,
            color=spec["color"],
            linewidth=2.3,
            marker=spec["marker"],
            markersize=4.5,
            markevery=max(1, len(xs) // 10),
            label=f"{spec['label']} (n={len(runs)})",
        )
        summary.append(f"- {spec['label']}: {len(runs)} seed(s)")

    ax.set_title(title)
    ax.set_xlabel("Environment Frames")
    ax.set_ylabel("Eval Episode Reward Mean")
    ax.set_ylim(bottom=-200, top=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return summary


def main() -> None:
    phase1_spread = {
        "mappo": {
            "label": "MAPPO",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_spread/mappo"),
            "color": "#1f77b4",
            "marker": "o",
        },
        "ippo": {
            "label": "IPPO",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_spread/ippo"),
            "color": "#ff7f0e",
            "marker": "s",
        },
        "vdn": {
            "label": "VDN",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_offpolicy_outputs/simple_spread/vdn"),
            "color": "#d62728",
            "marker": "^",
        },
        "iql": {
            "label": "IQL",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_iql_outputs/simple_spread/iql"),
            "color": "#9467bd",
            "marker": "D",
        },
    }

    phase1_reference = {
        "mappo": {
            "label": "MAPPO",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_reference/mappo"),
            "color": "#1f77b4",
            "marker": "o",
        },
        "ippo": {
            "label": "IPPO",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_reference/ippo"),
            "color": "#ff7f0e",
            "marker": "s",
        },
        "vdn": {
            "label": "VDN",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_offpolicy_outputs/simple_reference/vdn"),
            "color": "#d62728",
            "marker": "^",
        },
        "iql": {
            "label": "IQL",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_iql_outputs/simple_reference/iql"),
            "color": "#9467bd",
            "marker": "D",
        },
    }

    phase2_root = Path("/home/naikewu/RL/BenchMARL/phase2_mappo_ablation_outputs")
    phase2 = {
        "baseline": {
            "label": "baseline",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_spread/mappo"),
            "color": "#1f77b4",
            "marker": "o",
        },
        "clip_0p05": {
            "label": "clip=0.05",
            "path": phase2_root / "clip_epsilon" / "0p05",
            "color": "#2ca02c",
            "marker": "o",
        },
        "clip_0p1": {
            "label": "clip=0.1",
            "path": phase2_root / "clip_epsilon" / "0p1",
            "color": "#2ca02c",
            "marker": "s",
        },
        "iters_15": {
            "label": "iters=15",
            "path": phase2_root / "on_policy_n_minibatch_iters" / "15",
            "color": "#d62728",
            "marker": "o",
        },
        "iters_90": {
            "label": "iters=90",
            "path": phase2_root / "on_policy_n_minibatch_iters" / "90",
            "color": "#d62728",
            "marker": "s",
        },
        "minibatch_200": {
            "label": "minibatch=200",
            "path": phase2_root / "on_policy_minibatch_size" / "200",
            "color": "#9467bd",
            "marker": "o",
        },
        "minibatch_800": {
            "label": "minibatch=800",
            "path": phase2_root / "on_policy_minibatch_size" / "800",
            "color": "#9467bd",
            "marker": "s",
        },
        "frames_3000": {
            "label": "frames=3000",
            "path": phase2_root / "on_policy_collected_frames_per_batch" / "3000",
            "color": "#8c564b",
            "marker": "o",
        },
        "frames_12000": {
            "label": "frames=12000",
            "path": phase2_root / "on_policy_collected_frames_per_batch" / "12000",
            "color": "#8c564b",
            "marker": "s",
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5), dpi=160)
    summaries = []
    summaries.append(("Phase 1 - simple_spread", plot_panel(axes[0], phase1_spread, "Phase 1: simple_spread")))
    summaries.append(("Phase 1 - simple_reference", plot_panel(axes[1], phase1_reference, "Phase 1: simple_reference")))
    summaries.append(("Phase 2", plot_panel(axes[2], phase2, "Phase 2: MAPPO Ablations")))

    fig.tight_layout()
    output_dir = Path("/home/naikewu/RL/BenchMARL/plots/all_completed_three_panel")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "all_completed_three_panel.png", bbox_inches="tight")
    fig.savefig(output_dir / "all_completed_three_panel.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "all_completed_three_panel_summary.txt"
    with summary_path.open("w") as f:
        f.write("All Completed Results: Three-Panel Figure\n\n")
        f.write(f"Completion threshold: last_frames >= {COMPLETE_FRAMES}\n\n")
        for title, lines in summaries:
            f.write(f"{title}\n")
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
