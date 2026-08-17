"""生成实验总账表 docs/experiment_ledger.md。

扫描 result/ 下所有 full_result.json，逐条提取溯源字段：
result_id / git_commit / backend model / temperature / seed / iterations /
MC runs / case bank / HOPE 参数 / hardware / timestamp / output path。

字段缺失（旧结果在字段上线前生成）标为 "n/a"，并在总账头部说明。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "result"
OUT = ROOT / "docs" / "experiment_ledger.md"


def collect_results() -> list[dict]:
    rows = []
    for path in sorted(RESULT_DIR.rglob("full_result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = data.get("experiment_config") or {}
        manifest = data.get("runtime_manifest") or {}
        hw = manifest.get("hardware") or {}
        backend = manifest.get("llm_backend") or {}
        infer = manifest.get("llm_inference_params") or {}
        rows.append(
            {
                "output_path": str(path.relative_to(ROOT)),
                "result_id": data.get("result_id", "n/a"),
                "git_commit": (manifest.get("git_commit") or "n/a")[:10],
                "model": backend.get("model_id", "n/a"),
                "temperature": infer.get("temperature_generation", "n/a"),
                "seed": cfg.get("seed", "n/a"),
                "iterations": cfg.get("iterations", "n/a"),
                "mc_runs": cfg.get("sim_runs", "n/a"),
                "case_bank": str(cfg.get("case_bank_path", "n/a")).replace("data/", "").replace("data\\", ""),
                "sde_theta": cfg.get("sde_theta", "n/a"),
                "sde_epsilon": cfg.get("sde_epsilon", "n/a"),
                "hardware": f"{hw.get('gpu', 'n/a')} / {hw.get('cpu', 'n/a')}",
                "timestamp": manifest.get("collected_at", "n/a"),
                "fast_first": cfg.get("fast_first", False),
                "delta_refine": cfg.get("delta_refine", False),
            }
        )
    return rows


def main() -> None:
    rows = collect_results()
    header = [
        "# 实验总账表（Experiment Ledger）",
        "",
        "> 自动生成自 `result/**/full_result.json`。每行对应一次完整 `pipeline.run()` 执行。",
        "> **溯源说明**：`result_id` 与 `git_commit` 字段自 2026-08-17 起写入 `runtime_manifest`；",
        "> 此前的运行（ablation_suite_4090_rerun、ablation_4090_multiseed、scene_out_4090、",
        "> baselines_4090、fast_*、delta_* 等）生成于字段上线前，其 `result_id`/`git_commit` 标为 `n/a`，",
        "> 可通过 `timestamp` 与仓库 `git log` 交叉定位当时的代码状态；此后新运行将自动携带 `result_id` 与 `git_commit`。",
        "",
        "| # | result_id | git_commit | model | temp | seed | iters | MC | case_bank | HOPE(θ/ε) | hardware | timestamp | output_path |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, 1):
        header.append(
            f"| {i} | {r['result_id']} | {r['git_commit']} | {r['model']} | {r['temperature']} "
            f"| {r['seed']} | {r['iterations']} | {r['mc_runs']} | {r['case_bank']} "
            f"| {r['sde_theta']}/{r['sde_epsilon']} | {r['hardware']} | {r['timestamp']} | `{r['output_path']}` |"
        )
    header.append("")
    header.append(f"共 {len(rows)} 条运行记录。")

    OUT.write_text("\n".join(header), encoding="utf-8")
    print(f"saved: {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
