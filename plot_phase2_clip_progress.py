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


def plot(output_dir: Path) -> None:
    series = {
        "baseline": {
            "path": Path("/home/naikewu/RL/BenchMARL/phase1_outputs/simple_spread/mappo"),
            "color": "#1f77b4",
            "label": "baseline (clip=0.2)",
        },
        "clip_0.05": {
            "path": Path("/home/naikewu/RL/BenchMARL/phase2_mappo_ablation_outputs/clip_epsilon/0p05"),
            "color": "#2ca02c",
            "label": "clip=0.05",
        },
        "clip_0.1": {
            "path": Path("/home/naikewu/RL/BenchMARL/phase2_mappo_ablation_outputs/clip_epsilon/0p1"),
            "color": "#d62728",
            "label": "clip=0.1",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    panels = [
        ("train_curve", "Training Curve", "Collection Episode Reward Mean"),
        ("eval_curve", "Evaluation Curve", "Eval Episode Reward Mean"),
    ]

    for ax, (curve_key, title, ylabel) in zip(axes, panels):
        for _, spec in series.items():
            runs = load_runs(spec["path"])
            for run in runs:
                curve = run[curve_key]
                if not curve:
                    continue
                xs = [x for x, _ in curve]
                ys = [y for _, y in curve]
                ax.plot(xs, ys, color=spec["color"], alpha=0.28, linewidth=1.0)

            stats_curve = aggregate_stats(run[curve_key] for run in runs if run[curve_key])
            if stats_curve:
                xs = [x for x, _, _ in stats_curve]
                mean_ys = [y for _, y, _ in stats_curve]
                std_ys = [s for _, _, s in stats_curve]
                lower = [m - s for m, s in zip(mean_ys, std_ys)]
                upper = [m + s for m, s in zip(mean_ys, std_ys)]
                ax.fill_between(xs, lower, upper, color=spec["color"], alpha=0.16)
                ax.plot(
                    xs,
                    mean_ys,
                    color=spec["color"],
                    linewidth=2.8,
                    label=f"{spec['label']} (n={len(runs)})",
                )

        ax.set_title(f"Phase 2 MAPPO Clip Ablation - {title}")
        ax.set_xlabel("Environment Frames")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=-200, top=-25)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "phase2_clip_progress_zoomed.png", bbox_inches="tight")
    fig.savefig(output_dir / "phase2_clip_progress_zoomed.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        "Phase 2 MAPPO Clip Progress",
        "",
        "Included series:",
        "- baseline (Phase 1 mappo simple_spread, clip=0.2)",
        "- clip=0.05",
        "- clip=0.1",
        "",
        "Note:",
        "- clip=0.1 currently includes any available seed outputs, even if the last seed is still in progress.",
    ]
    (output_dir / "phase2_clip_progress_summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    plot(Path("/home/naikewu/RL/BenchMARL/plots/phase2_clip_progress"))
