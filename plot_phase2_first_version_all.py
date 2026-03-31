#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def build_frame_map(counters_path: Path) -> Dict[int, int]:
    return {idx: int(value) for idx, value in read_scalar_csv(counters_path)}


def to_frame_curve(scalar_path: Path, frame_map: Dict[int, int]) -> List[Tuple[int, float]]:
    curve: List[Tuple[int, float]] = []
    for idx, value in read_scalar_csv(scalar_path):
        if idx in frame_map:
            curve.append((frame_map[idx], value))
    return curve


def load_runs(seed_root: Path, allowed_seeds: set[str]) -> List[dict]:
    runs: List[dict] = []
    for seed_dir in sorted(p for p in seed_root.glob("seed_*") if p.is_dir()):
        seed_name = seed_dir.name.replace("seed_", "")
        if seed_name not in allowed_seeds:
            continue

        scalars_dir = pick_scalars_dir(seed_dir)
        if scalars_dir is None:
            continue

        counters_path = scalars_dir / "counters_total_frames.csv"
        train_path = scalars_dir / "collection_reward_episode_reward_mean.csv"
        eval_path = scalars_dir / "eval_reward_episode_reward_mean.csv"
        if not (counters_path.exists() and train_path.exists() and eval_path.exists()):
            continue

        frame_map = build_frame_map(counters_path)
        train_curve = to_frame_curve(train_path, frame_map)
        eval_curve = to_frame_curve(eval_path, frame_map)
        if not train_curve and not eval_curve:
            continue

        runs.append(
            {
                "seed": seed_name,
                "train_curve": train_curve,
                "eval_curve": eval_curve,
            }
        )
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


def discover_series() -> Dict[str, dict]:
    phase2_root = Path("/home/naikewu/RL/BenchMARL/phase2_mappo_ablation_outputs")
    return {
        "baseline": {
            "label": "baseline (clip=0.2)",
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


def plot(output_dir: Path) -> None:
    allowed_seeds = {"0", "1", "2"}
    series = discover_series()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=160)
    panels = [
        ("train_curve", "Training Curve", "Collection Episode Reward Mean"),
        ("eval_curve", "Evaluation Curve", "Eval Episode Reward Mean"),
    ]

    summary_lines = ["Phase 2 First Version Results", "", "Included series:"]

    for ax, (curve_key, title, ylabel) in zip(axes, panels):
        for _, spec in series.items():
            runs = load_runs(spec["path"], allowed_seeds)
            if not runs:
                continue

            for run in runs:
                curve = run[curve_key]
                if not curve:
                    continue
                xs = [x for x, _ in curve]
                ys = [y for _, y in curve]
                ax.plot(xs, ys, color=spec["color"], alpha=0.18, linewidth=0.9)

            stats_curve = aggregate_stats(run[curve_key] for run in runs if run[curve_key])
            if not stats_curve:
                continue

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
                markersize=5,
                markevery=max(1, len(xs) // 10),
                label=f"{spec['label']} (n={len(runs)})",
            )

            if curve_key == "eval_curve":
                summary_lines.append(f"- {spec['label']}: {len(runs)} seed(s)")

        ax.set_title(f"Phase 2 First Version - {title}")
        ax.set_xlabel("Environment Frames")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=-200, top=-25)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase2_first_version_all.png", bbox_inches="tight")
    fig.savefig(output_dir / "phase2_first_version_all.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_lines.append("")
    summary_lines.append("Note: this figure uses only the original Phase 2 seeds 0,1,2 and includes all completed first-version ablations.")
    (output_dir / "phase2_first_version_all_summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    plot(Path("/home/naikewu/RL/BenchMARL/plots/phase2_first_version_all"))
