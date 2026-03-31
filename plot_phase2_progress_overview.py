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


def load_runs(seed_root: Path) -> List[dict]:
    runs: List[dict] = []
    for seed_dir in sorted(p for p in seed_root.glob("seed_*") if p.is_dir()):
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
                "seed": seed_dir.name.replace("seed_", ""),
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
    series: Dict[str, dict] = {
        "baseline": {
            "label": "baseline (clip=0.2)",
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_spread/mappo"),
            "color": "#1f77b4",
            "marker": "o",
        }
    }

    phase2_root = Path("/home/naikewu/RL/BenchMARL/phase2_mappo_ablation_outputs")
    group_colors = {
        "clip_epsilon": "#2ca02c",
        "on_policy_n_minibatch_iters": "#d62728",
        "on_policy_minibatch_size": "#9467bd",
        "on_policy_collected_frames_per_batch": "#8c564b",
    }
    pretty_names = {
        "clip_epsilon": "clip",
        "on_policy_n_minibatch_iters": "iters",
        "on_policy_minibatch_size": "minibatch",
        "on_policy_collected_frames_per_batch": "frames",
    }
    markers = ["o", "s", "^", "D", "v", "P"]

    for group_dir in sorted(p for p in phase2_root.iterdir() if p.is_dir()):
        value_dirs = sorted(p for p in group_dir.iterdir() if p.is_dir())
        for idx, value_dir in enumerate(value_dirs):
            key = f"{group_dir.name}/{value_dir.name}"
            series[key] = {
                "label": f"{pretty_names.get(group_dir.name, group_dir.name)}={value_dir.name}",
                "path": value_dir,
                "color": group_colors.get(group_dir.name, "#7f7f7f"),
                "marker": markers[idx % len(markers)],
            }
    return series


def plot(output_dir: Path) -> None:
    series = discover_series()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=160)
    panels = [
        ("train_curve", "Training Curve", "Collection Episode Reward Mean"),
        ("eval_curve", "Evaluation Curve", "Eval Episode Reward Mean"),
    ]

    plotted_summary: List[str] = []
    for ax, (curve_key, title, ylabel) in zip(axes, panels):
        for _, spec in series.items():
            runs = load_runs(spec["path"])
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
                plotted_summary.append(f"- {spec['label']}: {len(runs)} seed(s)")

        ax.set_title(f"Phase 2 Progress - {title}")
        ax.set_xlabel("Environment Frames")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=-200, top=-25)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase2_progress_overview.png", bbox_inches="tight")
    fig.savefig(output_dir / "phase2_progress_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = ["Phase 2 Progress Overview", "", "Included series:"]
    summary.extend(plotted_summary)
    summary.append("")
    summary.append("Note: n counts only seeds with scalar CSVs currently present on disk.")
    (output_dir / "phase2_progress_overview_summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    plot(Path("/home/naikewu/RL/BenchMARL/plots/phase2_progress_overview"))
