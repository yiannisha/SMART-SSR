from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ssr_metrics import collect_run_metrics


def parse_run_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        label = Path(spec).name
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid run label in spec: {spec}")
    if not path.exists():
        raise FileNotFoundError(f"Run path does not exist: {path}")
    return label, path


def format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_summary_csv(output_path: Path, runs: List[Dict[str, object]]) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "run_path", "average_rouge_l", "bwt", "fwt"],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run["label"],
                    "run_path": str(run["path"]),
                    "average_rouge_l": format_metric(run["average_rouge_l"]),
                    "bwt": format_metric(run["bwt"]),
                    "fwt": format_metric(run["fwt"]),
                }
            )


def add_bar_labels(ax: plt.Axes, bars, values: List[float | None]) -> None:
    for bar, value in zip(bars, values):
        x_pos = bar.get_x() + bar.get_width() / 2
        if value is None:
            ax.text(x_pos, 0.0, "N/A", ha="center", va="bottom", fontsize=9)
            continue
        offset = 0.6 if value >= 0 else -0.6
        va = "bottom" if value >= 0 else "top"
        ax.text(x_pos, value + offset, f"{value:.2f}", ha="center", va=va, fontsize=9)


def plot_summary_metrics(output_path: Path, runs: List[Dict[str, object]]) -> None:
    labels = [str(run["label"]) for run in runs]
    metric_specs = [
        ("average_rouge_l", "Final Avg Rouge-L"),
        ("bwt", "BWT"),
        ("fwt", "FWT"),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(15, 4.8))
    if len(metric_specs) == 1:
        axes = [axes]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        raw_values = [run[metric_key] for run in runs]
        plot_values = [0.0 if value is None else float(value) for value in raw_values]
        bars = ax.bar(labels, plot_values, color="#4C72B0", alpha=0.9)
        for bar, value in zip(bars, raw_values):
            if value is None:
                bar.set_facecolor("#C9CED6")
                bar.set_hatch("//")
        ax.axhline(0.0, color="#222222", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        add_bar_labels(ax, bars, raw_values)

        finite_values = [float(value) for value in raw_values if value is not None]
        if finite_values:
            min_value = min(finite_values + [0.0])
            max_value = max(finite_values + [0.0])
            span = max(max_value - min_value, 1.0)
            ax.set_ylim(min_value - 0.2 * span, max_value + 0.25 * span)
        else:
            ax.set_ylim(-1.0, 1.0)

    fig.suptitle("SSR Run Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(output_path: Path, tasks: List[str], runs: List[Dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x_positions = list(range(1, len(tasks) + 1))
    x_labels = [f"{idx}:{task}" for idx, task in enumerate(tasks, start=1)]

    for run in runs:
        stage_curve = run["per_stage_average_rouge_l_seen_tasks"]
        y_values = [
            math.nan if stage_curve.get(task) is None else float(stage_curve[task])
            for task in tasks
        ]
        ax.plot(x_positions, y_values, marker="o", linewidth=2, label=str(run["label"]))

    ax.set_xticks(x_positions, x_labels)
    ax.set_xlabel("SSR iteration / task")
    ax.set_ylabel("Average Rouge-L on seen tasks")
    ax.set_title("Rouge-L Trajectory Across SSR Iterations")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form label=/path/to/run or /path/to/run.",
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, object]] = []
    reference_tasks: List[str] | None = list(args.tasks) if args.tasks else None

    for spec in args.run:
        label, path = parse_run_spec(spec)
        metrics = collect_run_metrics(path, reference_tasks)
        tasks = list(metrics["tasks"])
        if reference_tasks is None:
            reference_tasks = tasks
        elif tasks != reference_tasks:
            raise ValueError(
                f"Task mismatch for run {label}: expected {reference_tasks}, got {tasks}"
            )
        runs.append(
            {
                "label": label,
                "path": path,
                **metrics,
            }
        )

    if reference_tasks is None:
        raise ValueError("No runs were loaded.")

    write_summary_csv(output_dir / "comparison_summary.csv", runs)
    plot_summary_metrics(output_dir / "summary_metrics.png", runs)
    plot_trajectory(output_dir / "average_rouge_l_trajectory.png", reference_tasks, runs)

    details_path = output_dir / "comparison_details.json"
    details_path.write_text(
        json.dumps(
            {
                "tasks": reference_tasks,
                "runs": [
                    {
                        "label": run["label"],
                        "path": str(run["path"]),
                        "average_rouge_l": run["average_rouge_l"],
                        "bwt": run["bwt"],
                        "fwt": run["fwt"],
                        "per_stage_average_rouge_l_seen_tasks": run[
                            "per_stage_average_rouge_l_seen_tasks"
                        ],
                        "per_stage_average_rouge_l_all_tasks": run[
                            "per_stage_average_rouge_l_all_tasks"
                        ],
                        "score_matrix_rouge_l": run["score_matrix_rouge_l"],
                    }
                    for run in runs
                ],
            },
            indent=2,
        )
    )

    print(f"Saved summary table to {output_dir / 'comparison_summary.csv'}")
    print(f"Saved metric bars to {output_dir / 'summary_metrics.png'}")
    print(f"Saved trajectory plot to {output_dir / 'average_rouge_l_trajectory.png'}")
    print(f"Saved detailed metrics to {details_path}")


if __name__ == "__main__":
    main()
