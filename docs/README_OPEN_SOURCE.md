# memento-hope — Open-Source Task Sandbox

> Source for *A Dual-System Enhanced Large Language Model Architecture for Real-Time
> Joint-Operation Mission Planning in Command and Control Systems* (submitted to
> *Defence Technology*).

This repository releases the **lightweight task sandbox** used as the algorithmic
objective evaluator in the paper. It is deliberately not a physics-based
simulation engine (e.g., OneSAF, FLAMES); it is a closed-form, deterministic
five-stage kill-chain evaluator that provides a reproducible reward/diagnosis
signal for the LLM planning loop. All stage scores, metrics, and ablations in the
paper are produced by this code.

## Requirements

- Python 3.11+
- See `requirements.txt` / `uv.lock`. Key dependencies: `openai`, `torch`,
  `numpy`, `sentence-transformers` (optional FAISS-backed retrieval),
  `faiss-cpu` (optional), `loguru`, `pytest`.

```bash
python -m pip install -r requirements.txt
# or: uv sync
```

## Five-stage evaluation (formal definition)

The evaluator implements the closed-form stage score:

```
s_k = clamp( sigmoid( (g_k / lambda_k) - pi_k ) * (0.70 + 0.35 * phi_k) + rho * u_k , 0.12, 0.96 )
g_k = sum_j alpha_{k,j} * (blue_{p,j} * w_j / max(bluebar_j, eps))   - sum_j beta_{k,j} * red_{r,j}
```

where `g_k` is the capability-weighted adversary gap, `pi_k` the scenario penalty
(terrain/weather/civilian/EW), `phi_k` the stage-fit modulation, and `u_k` the
stage-relevant action signal. The implementation lives in
`military_research/engine.py` in the `PlanSimulator` class:
`_evaluate_kill_chain`, `_phase_requirements`, `_stage_signal_from_actions`,
`_bounded_score`; the domain types are in `military_research/domain.py`.

## Run the tests (one command)

```bash
python -m pytest tests -q
```

The suite covers the domain model, the evaluator, memory retrieval, the plan-repair
logic, and the DoDAF/C2SIM exporters.

## Running the pipeline (single scenario)

```bash
# local OpenAI-compatible backend (e.g., LM Studio)
$env:LOCAL_LLM_BASE_URL="http://localhost:1234/v1"
$env:LOCAL_LLM_API_KEY="lm-studio"
$env:LOCAL_LLM_MODEL="deepseek-r1-0528-qwen3-8b"

python -X utf8 -m military_research.cli \
  --scenario data/sample_joint_operation.json \
  --case-bank data/military_case_bank_frozen_seed.jsonl \
  --iterations 4 --sim-runs 20 --seed 7 --disable-writeback \
  --output-dir result/demo_run
```

To reproduce the dual-system real-time path (System 1 emergency + System 2 delta
patching), add `--fast-first --delta-refine`.

## License

This repository is released under the MIT License (see `LICENSE`). The research is
intended for academic and simulation-based study only; it does not constitute real
operational command, targeting, or weapon employment advice.
