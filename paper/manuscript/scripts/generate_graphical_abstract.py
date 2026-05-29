from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = ROOT / "paper" / "manuscript" / "figures"
PNG_PATH = FIGURE_DIR / "graphical_abstract_smpt.png"
PDF_PATH = FIGURE_DIR / "graphical_abstract_smpt.pdf"
TIFF_PATH = FIGURE_DIR / "graphical_abstract_smpt.tiff"


def add_stage(ax, x: float, width: float, title: str, body: str, face: str, hatch: str = "") -> None:
    box = Rectangle(
        (x, 0.26),
        width,
        0.48,
        facecolor=face,
        edgecolor="black",
        linewidth=1.2,
        hatch=hatch,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        0.64,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        0.44,
        body,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.2,
        wrap=True,
    )


def add_arrow(ax, x0: float, x1: float) -> None:
    arrow = FancyArrowPatch(
        (x0, 0.50),
        (x1, 0.50),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color="black",
    )
    ax.add_patch(arrow)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.92,
        "Structured memory + adaptive weighting + reflection improve mission success across 5/5 held-out scenarios",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    width = 0.135
    xs = [0.03, 0.195, 0.36, 0.525, 0.69, 0.855]
    fills = ["#f2f2f2", "#dddddd", "#f2f2f2", "#dddddd", "#f2f2f2", "#dddddd"]
    hatches = ["", "//", "", "//", "", "//"]
    titles = [
        "Scenario input",
        "JSONL memory",
        "HOPE modulation",
        "Structured LLM plan",
        "Simulation + MC",
        "Export + gain",
    ]
    bodies = [
        "Mission type\nTerrain, weather\nThreat, EW, time",
        "CaseBank JSONL\nFAISS when available\nKeyword fallback",
        "Scenario encoder\nFast/slow capability\nbounded fusion",
        "Typed phases\nAction allocation\nReflection patch",
        "Five-stage scores\n50-run summaries\nbottleneck feedback",
        "DoDAF OV-5b/OV-6c\nC2SIM order\n5/5 scene-out gains",
    ]

    for x, face, hatch, title, body in zip(xs, fills, hatches, titles, bodies):
        add_stage(ax, x, width, title, body, face, hatch)

    for idx in range(len(xs) - 1):
        add_arrow(ax, xs[idx] + width, xs[idx + 1] - 0.01)

    ax.text(
        0.442,
        0.18,
        "Dashed hatching marks auxiliary modules or derived artifacts; all elements remain legible in grayscale print.",
        ha="center",
        va="center",
        fontsize=9.5,
    )
    ax.text(
        0.77,
        0.18,
        "Non-generative artwork prepared for separate SMPT submission.",
        ha="center",
        va="center",
        fontsize=9.5,
    )

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(TIFF_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
