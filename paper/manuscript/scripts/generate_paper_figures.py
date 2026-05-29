import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "paper" / "manuscript" / "figures"

ABLA_DIR = ROOT / "result" / "ablation_suite_seed7_v3"
ITER20_DIR = ROOT / "result" / "generalization_suite_seed7_v2_sceneout_iter20"
ITER50_DIR = ROOT / "result" / "generalization_suite_seed7_v2_sceneout_iter50"

SCENES = [
    ("coastal_joint_assault", "Coastal"),
    ("island_resupply_corridor", "Island"),
    ("mountain_corridor_recon", "Mountain"),
    ("river_crossing_breakthrough", "River"),
    ("urban_hub_defense", "Urban"),
]

GROUPS = [
    ("ablation_pure_llm", "Pure LLM", "#6b7280"),
    ("ablation_memento", "LLM + Memento", "#3b82f6"),
    ("ablation_hope", "LLM + Hope", "#f59e0b"),
    ("ablation_reflection_only", "LLM + Reflection", "#8b5cf6"),
    ("ablation_full", "Full", "#10b981"),
]

GROUP_SHORT_LABELS = ["Pure", "Memento", "Hope", "Reflect.", "Full"]

ABLA_GROUPS = [
    ("ablation_pure_llm", "Pure LLM", "#6b7280"),
    ("ablation_memento", "LLM + Memento", "#3b82f6"),
    ("ablation_hope", "LLM + Hope", "#f59e0b"),
    ("ablation_reflection_only", "LLM + Reflection", "#8b5cf6"),
    ("ablation_full", "Full", "#10b981"),
]

TREND_METRICS = [
    ("mission_success", "Mission success"),
    ("overall_effectiveness", "Overall effectiveness"),
    ("ler", "LER"),
    ("completion_time_hours", "Completion time (h)"),
    ("command_resilience", "Command resilience"),
]

STAGES = [
    ("detect", "Detect"),
    ("disrupt", "Disrupt"),
    ("breach", "Breach"),
    ("control", "Control"),
    ("sustain", "Sustain"),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_bar_data() -> list[dict]:
    rows = []
    for scene_key, scene_label in SCENES:
        for group_key, group_label, _ in GROUPS:
            result_path = ITER20_DIR / scene_key / group_key / "full_result.json"
            data = load_json(result_path)
            rows.append(
                {
                    "scene_key": scene_key,
                    "scene_label": scene_label,
                    "group_key": group_key,
                    "group_label": group_label,
                    "mission_success": data["best_simulation"]["mission_success"],
                }
            )

    csv_path = FIG_DIR / "generalization_scene_mission_success.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scene_key",
                "scene_label",
                "group_key",
                "group_label",
                "mission_success",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def export_ablation_stage_data() -> list[dict]:
    rows = []
    reflection_dir = ROOT / "result" / "ablation_reflection_only_seed7"
    for group_key, group_label, color in ABLA_GROUPS:
        result_path = (
            reflection_dir / "full_result.json"
            if group_key == "ablation_reflection_only"
            else ABLA_DIR / group_key / "full_result.json"
        )
        data = load_json(result_path)
        stage_scores = data["best_simulation"]["stage_scores"]
        for stage_key, stage_label in STAGES:
            rows.append(
                {
                    "group_key": group_key,
                    "group_label": group_label,
                    "color": color,
                    "stage_key": stage_key,
                    "stage_label": stage_label,
                    "stage_score": stage_scores[stage_key],
                }
            )

    csv_path = FIG_DIR / "ablation_stage_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_key",
                "group_label",
                "color",
                "stage_key",
                "stage_label",
                "stage_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_ablation_stage_figure(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.1))
    stage_labels = [label for _, label in STAGES]
    x = np.arange(len(stage_labels))
    width = 0.15

    full_values = None
    for idx, (group_key, group_label, color) in enumerate(ABLA_GROUPS):
        group_rows = [row for row in rows if row["group_key"] == group_key]
        values = [row["stage_score"] for row in group_rows]
        if group_key == "ablation_full":
            full_values = values
        offset = (idx - ((len(ABLA_GROUPS) - 1) / 2)) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=group_label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    y_values = [row["stage_score"] for row in rows]
    y_min = max(0.0, min(y_values) - 0.025)
    y_max = min(1.0, max(y_values) + 0.025)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=11)
    ax.set_ylabel("Stage score", fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.legend(ncol=5, fontsize=8.5, frameon=False, loc="upper left")

    if full_values is not None:
        weakest_idx = int(np.argmin(full_values))
        weakest_label = stage_labels[weakest_idx]
        weakest_value = full_values[weakest_idx]
        ax.annotate(
            f"Full bottleneck: {weakest_label.lower()}",
            xy=(x[weakest_idx] + (((len(ABLA_GROUPS) - 1) / 2) * width), weakest_value),
            xytext=(x[weakest_idx] + 0.55, weakest_value + 0.035),
            textcoords="data",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#111827"},
            ha="left",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_stage_profile.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "ablation_stage_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def make_bar_figure(rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 9.6))
    axes = axes.flatten()
    colors = [color for _, _, color in GROUPS]

    for axis, (scene_key, scene_label) in zip(axes, SCENES):
        scene_rows = [row for row in rows if row["scene_key"] == scene_key]
        values = [row["mission_success"] for row in scene_rows]
        y_min = max(0.0, min(values) - 0.015)
        y_max = min(1.0, max(values) + 0.015)
        ticks = np.linspace(y_min, y_max, 5)
        bars = axis.bar(
            GROUP_SHORT_LABELS,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        axis.set_title(f"{scene_label} scenario", fontsize=13)
        axis.set_ylim(y_min, y_max)
        axis.set_yticks(ticks)
        axis.set_ylabel("Mission success", fontsize=11)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        axis.tick_params(axis="x", rotation=12, labelsize=10)
        axis.tick_params(axis="y", labelsize=10)
        axis.text(
            0.03,
            0.97,
            f"y-range: {y_min:.2f}-{y_max:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#374151",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (y_max - y_min) * 0.04,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[-1].axis("off")
    fig.suptitle("Mission success by ablation setting within each scene-out scenario", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "generalization_scene_group_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "generalization_scene_group_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def export_trend_data() -> list[dict]:
    scene_histories = []
    for scene_key, _ in SCENES:
        result_path = ITER50_DIR / scene_key / "ablation_full" / "full_result.json"
        data = load_json(result_path)
        scene_histories.append(data["optimization_history"])

    rows = []
    for idx in range(len(scene_histories[0])):
        row = {"iteration": scene_histories[0][idx]["iteration"]}
        for metric_key, _ in TREND_METRICS:
            values = [history[idx]["simulation"][metric_key] for history in scene_histories]
            row[metric_key] = float(sum(values) / len(values))
        rows.append(row)

    csv_path = FIG_DIR / "full_iter50_average_trends.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration"] + [metric_key for metric_key, _ in TREND_METRICS],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def centered_moving_average(values: list[float], window: int = 5) -> list[float]:
    smoothed = []
    radius = window // 2
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        smoothed.append(float(sum(values[lo:hi]) / (hi - lo)))
    return smoothed


def export_smoothed_trend_data(rows: list[dict]) -> list[dict]:
    smoothed_rows = [{"iteration": row["iteration"]} for row in rows]
    for metric_key, _ in TREND_METRICS:
        raw_values = [row[metric_key] for row in rows]
        smooth_values = centered_moving_average(raw_values, window=5)
        for row, smooth_value in zip(smoothed_rows, smooth_values):
            row[metric_key] = smooth_value

    csv_path = FIG_DIR / "full_iter50_average_trends_smoothed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration"] + [metric_key for metric_key, _ in TREND_METRICS],
        )
        writer.writeheader()
        writer.writerows(smoothed_rows)
    return smoothed_rows


def make_trend_figure(rows: list[dict]) -> None:
    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 9.8))
    axes = axes.flatten()
    colors = ["#2563eb", "#0f766e", "#d97706", "#9333ea", "#dc2626"]

    for axis, (metric_key, label), color in zip(axes, TREND_METRICS, colors):
        values = [row[metric_key] for row in rows]
        smooth_values = centered_moving_average(values, window=5)
        span = max(smooth_values) - min(smooth_values)
        padding = max(span * 0.18, 0.005)
        axis.plot(iterations, values, color=color, linewidth=1.0, alpha=0.28)
        axis.plot(iterations, smooth_values, color=color, linewidth=2.4)
        axis.set_title(label, fontsize=13)
        axis.set_xlabel("Iteration", fontsize=11)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axis.set_xlim(min(iterations), max(iterations))
        axis.set_ylim(min(smooth_values) - padding, max(smooth_values) + padding)
        axis.tick_params(axis="both", labelsize=10)

    axes[-1].axis("off")
    fig.suptitle("Full-pipeline average metric trends over 50 planning iterations", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "full_iter50_average_trends.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "full_iter50_average_trends.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript figure assets.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["ablation", "generalization", "trend"],
        help="Generate only the selected figure groups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = set(args.only or ["ablation", "generalization", "trend"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if "ablation" in selected:
        ablation_stage_rows = export_ablation_stage_data()
        make_ablation_stage_figure(ablation_stage_rows)
    if "generalization" in selected:
        bar_rows = export_bar_data()
        make_bar_figure(bar_rows)
    if "trend" in selected:
        trend_rows = export_trend_data()
        export_smoothed_trend_data(trend_rows)
        make_trend_figure(trend_rows)


if __name__ == "__main__":
    main()
