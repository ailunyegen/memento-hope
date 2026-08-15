"""固化论文图表资产：全量重写退化 vs 增量单调收敛 对比图。

数据源：
- 曲线 A（全量重写，Fast-10）：result/fast10_benchmark/full_result.json
  -> optimization_history[*].simulation.mission_success
- 曲线 B（Delta Loop v2，8 轮）：result/delta_loop_v2.json -> history[*].best_ms
- 曲线 C（Delta Loop 15 轮压榨，可选）：result/delta_15_squeeze/full_result.json
  -> delta_history[*].best_ms

输出：result/figures/full_rewrite_vs_delta.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent

# 中文字体（Windows 环境）
for name in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False


def load_fast10_trajectory() -> list[float]:
    path = ROOT / "result/fast10_benchmark/full_result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [h["simulation"]["mission_success"] for h in data["optimization_history"]]


def load_delta_v2_trajectory() -> list[float]:
    path = ROOT / "result/delta_loop_v2.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [h["best_ms"] for h in data["history"]]


def load_delta_15_trajectory() -> list[float] | None:
    path = ROOT / "result/delta_15_squeeze/full_result.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    hist = data.get("delta_history")
    if not hist:
        return None
    return [h.get("best_ms", 0.0) for h in hist]


def main() -> None:
    fast10 = load_fast10_trajectory()
    delta_v2 = load_delta_v2_trajectory()
    delta_15 = load_delta_15_trajectory()

    fig, ax = plt.subplots(figsize=(9, 5.2))

    # 曲线 A：全量重写（Fast-10）
    ax.plot(
        range(1, len(fast10) + 1), fast10,
        color="#d9534f", marker="o", markersize=5, linewidth=1.8,
        label=f"Full Rewrite / Single-Candidate (Fast-10), peak={max(fast10):.4f}",
    )
    # 曲线 B：Delta Loop v2（8 轮）
    ax.plot(
        range(1, len(delta_v2) + 1), delta_v2,
        color="#337ab7", marker="s", markersize=6, linewidth=2.2,
        label=f"Delta Loop v2 (8 rounds), final={delta_v2[-1]:.4f}",
    )
    # 曲线 C：Delta 15 轮压榨（可选）
    if delta_15:
        ax.plot(
            range(1, len(delta_15) + 1), delta_15,
            color="#5cb85c", marker="^", markersize=6, linewidth=2.0, linestyle="--",
            label=f"Delta Loop 15-round squeeze, final={delta_15[-1]:.4f}",
        )

    # 参考线
    ax.axhline(0.6176, color="#999999", linestyle=":", linewidth=1.2)
    ax.text(len(fast10) + 0.2, 0.6178, "Emergency anchor (System 1)", fontsize=8, color="#666666")
    ax.axhline(0.6347, color="#f0ad4e", linestyle="-.", linewidth=1.2)
    ax.text(len(fast10) + 0.2, 0.6352, "Fast-10 peak", fontsize=8, color="#b07c1e")
    ax.axhline(0.6388, color="#7a4fb0", linestyle="-.", linewidth=1.2)
    ax.text(len(fast10) + 0.2, 0.6393, "Full-10 (paper)", fontsize=8, color="#7a4fb0")

    ax.set_xlabel("Optimization iteration", fontsize=11)
    ax.set_ylabel("Mission success (50-MC)", fontsize=11)
    ax.set_title(
        "Full Rewrite Regression vs Delta Patching Monotonic Convergence",
        fontsize=12,
    )
    ax.set_ylim(0.52, 0.66)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = ROOT / "result/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_rewrite_vs_delta.png"
    fig.savefig(out_path, dpi=200)
    print(f"chart saved: {out_path}")

    # 随图输出数据摘要
    summary = {
        "full_rewrite_fast10": fast10,
        "delta_v2_best_ms": delta_v2,
        "delta_15_best_ms": delta_15,
        "references": {"emergency": 0.6176, "fast10_peak": 0.6347, "full10_paper": 0.6388},
    }
    (out_dir / "full_rewrite_vs_delta_data.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"data saved: {out_dir / 'full_rewrite_vs_delta_data.json'}")


if __name__ == "__main__":
    main()
