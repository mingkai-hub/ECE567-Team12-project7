#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def build_frame_map(counters_path: Path) -> Dict[int, int]:
    return {idx: int(value) for idx, value in read_scalar_csv(counters_path)}


def to_frame_curve(
    scalar_path: Path,
    frame_map: Dict[int, int],
) -> List[Tuple[int, float]]:
    curve: List[Tuple[int, float]] = []
    for idx, value in read_scalar_csv(scalar_path):
        if idx in frame_map:
            curve.append((frame_map[idx], value))
    return curve


def pick_scalars_dir(seed_dir: Path) -> Path | None:
    candidates = sorted(seed_dir.rglob("scalars"))
    return candidates[-1] if candidates else None


def aggregate_mean(curves: Iterable[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for curve in curves:
        for frame, value in curve:
            buckets[frame].append(value)
    return [(frame, sum(vals) / len(vals)) for frame, vals in sorted(buckets.items())]


def aggregate_stats(
    curves: Iterable[List[Tuple[int, float]]]
) -> List[Tuple[int, float, float]]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for curve in curves:
        for frame, value in curve:
            buckets[frame].append(value)

    stats: List[Tuple[int, float, float]] = []
    for frame, vals in sorted(buckets.items()):
        arr = np.asarray(vals, dtype=float)
        stats.append((frame, float(arr.mean()), float(arr.std(ddof=0))))
    return stats


def discover_runs(root: Path) -> Dict[str, Dict[str, List[dict]]]:
    data: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        task_name = task_dir.name
        for alg_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            alg_name = alg_dir.name
            for seed_dir in sorted(p for p in alg_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
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

                data[task_name][alg_name].append(
                    {
                        "seed": seed_dir.name.replace("seed_", ""),
                        "seed_dir": seed_dir,
                        "scalars_dir": scalars_dir,
                        "train_curve": train_curve,
                        "eval_curve": eval_curve,
                    }
                )

    return data


def merge_run_data(
    all_data: Iterable[Dict[str, Dict[str, List[dict]]]]
) -> Dict[str, Dict[str, List[dict]]]:
    merged: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for data in all_data:
        for task_name, alg_runs in data.items():
            for alg_name, runs in alg_runs.items():
                merged[task_name][alg_name].extend(runs)
    return merged


def plot_task(
    task: str,
    alg_runs: Dict[str, List[dict]],
    output_dir: Path,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    colors = {
        "mappo": "#1f77b4",
        "ippo": "#ff7f0e",
        "qmix": "#2ca02c",
        "vdn": "#d62728",
        "iql": "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    titles = [
        ("train_curve", "Training Curve", "Collection Episode Reward Mean"),
        ("eval_curve", "Evaluation Curve", "Eval Episode Reward Mean"),
    ]

    for ax, (curve_key, title, ylabel) in zip(axes, titles):
        for alg_name, runs in sorted(alg_runs.items()):
            color = colors.get(alg_name, None)

            for run in runs:
                curve = run[curve_key]
                if not curve:
                    continue
                xs = [x for x, _ in curve]
                ys = [y for _, y in curve]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    alpha=0.28,
                    linewidth=1.0,
                    label=f"{alg_name} seed {run['seed']}" if False else None,
                )

            stats_curve = aggregate_stats(run[curve_key] for run in runs if run[curve_key])
            if stats_curve:
                xs = [x for x, _, _ in stats_curve]
                mean_ys = [y for _, y, _ in stats_curve]
                std_ys = [s for _, _, s in stats_curve]
                lower = [m - s for m, s in zip(mean_ys, std_ys)]
                upper = [m + s for m, s in zip(mean_ys, std_ys)]

                ax.fill_between(xs, lower, upper, color=color, alpha=0.16)
                ax.plot(
                    xs,
                    mean_ys,
                    color=color,
                    linewidth=2.8,
                    label=f"{alg_name} mean +/- std (n={len(runs)})",
                )

        ax.set_title(f"{task} - {title}")
        ax.set_xlabel("Environment Frames")
        ax.set_ylabel(ylabel)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Phase 1 Curves: {task}", fontsize=14)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{task}_train_eval_curves.png"
    pdf_path = output_dir / f"{task}_train_eval_curves.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(data: Dict[str, Dict[str, List[dict]]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase1_plot_summary.txt"
    with summary_path.open("w") as f:
        f.write("Phase 1 Plot Summary\n\n")
        for task, alg_runs in sorted(data.items()):
            f.write(f"Task: {task}\n")
            for alg_name, runs in sorted(alg_runs.items()):
                seeds = ", ".join(sorted(run["seed"] for run in runs))
                f.write(f"- {alg_name}: {len(runs)} runs (seeds: {seeds})\n")
            f.write("\n")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BenchMARL Phase 1 training/eval curves.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/naikewu/RL/BenchMARL/phase1_outputs"),
        help="Root directory that contains task/algorithm/seed outputs.",
    )
    parser.add_argument(
        "--extra-root",
        type=Path,
        action="append",
        default=[],
        help="Additional root directory to merge into the plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/naikewu/RL/BenchMARL/plots/phase1_curves"),
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional single task to plot, e.g. simple_spread.",
    )
    parser.add_argument(
        "--include-alg",
        type=str,
        action="append",
        default=[],
        help="Algorithm name to include. Repeat this flag to include multiple algorithms.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional lower bound for the y-axis.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional upper bound for the y-axis.",
    )
    args = parser.parse_args()

    roots = [args.root, *args.extra_root]
    data = merge_run_data(discover_runs(root) for root in roots if root.exists())
    if args.task is not None:
        data = {k: v for k, v in data.items() if k == args.task}
    if args.include_alg:
        wanted = set(args.include_alg)
        data = {
            task: {alg: runs for alg, runs in alg_runs.items() if alg in wanted}
            for task, alg_runs in data.items()
        }
        data = {task: alg_runs for task, alg_runs in data.items() if alg_runs}

    if not data:
        raise SystemExit("No completed runs with scalar CSVs were found.")

    summary_path = write_summary(data, args.output_dir)
    print(f"Wrote summary: {summary_path}")

    for task, alg_runs in sorted(data.items()):
        plot_task(task, alg_runs, args.output_dir, ymin=args.ymin, ymax=args.ymax)
        total_runs = sum(len(runs) for runs in alg_runs.values())
        print(f"Plotted {task}: {total_runs} available runs across {len(alg_runs)} algorithms.")


if __name__ == "__main__":
    main()
