# Evidence Inventory

This note records the main claims used in the manuscript and the repository-local files that support them.

## Canonical Evidence Scope

- Single-scenario ablation suite:
  - `result/ablation_suite_seed7_v3/`
- Cross-scenario generalization suite:
  - `result/generalization_suite_seed7_v2_sceneout_iter20/`
- Auxiliary figure-only evidence:
  - `result/generalization_suite_seed7_v2_sceneout_iter50/`
- Supplemental reflection-only ablation:
  - `result/ablation_reflection_only_seed7/`
- Scenario and case-bank metadata:
  - `data/sample_joint_operation.json`
  - `data/generalization_scenarios/manifest.json`
  - `data/generalization_case_banks_v2_sceneout/summary.json`
- Core implementation:
  - `military_research/domain.py`
  - `military_research/case_memory.py`
  - `military_research/engine.py`
  - `military_research/exporters.py`
  - `run_ablation_suite.ps1`
  - `run_reflection_only_ablation.ps1`
  - `run_ablation_multiseed.ps1`
  - `military_research/summarize_multiseed_ablation.py`
  - `run_generalization_suite.ps1`
- Test coverage used as implementation support:
  - `tests/test_case_memory.py`
  - `tests/test_simulator.py`
  - `tests/test_exporters.py`
  - `tests/test_runtime_manifest.py`
- Local reference list:
  - repository-root reference list text file
- Author-supplied metadata confirmed on 2026-05-16 and cross-checked against refreshed rerun manifests on 2026-05-23:
  - authors: `Changrui Zhang`; `Chengwei Yang`
  - affiliation: `Beijing Institute of Technology`
  - corresponding author: `Chengwei Yang`
  - email: `yangchengwei@bit.edu.cn`
  - funding: none
  - backend model: `deepseek-r1-0528-qwen3-8b`
  - Python virtual environment: `Memento`
  - CPU: `13th Gen Intel(R) Core(TM) i7-13700F`
  - GPU: `NVIDIA GeForce RTX 3060`

## Manuscript-to-Citation Mapping

| Manuscript location | Purpose | BibTeX keys |
| --- | --- | --- |
| `sections/introduction.tex`, paragraph 1 | LLM planning context | `brown2020llm` |
| `sections/introduction.tex`, paragraph 2 | Retrieval, sentence embeddings, and case-based reasoning context | `aamodt1994cbr`, `kolodner1993cbr`, `lewis2020rag`, `reimers2019sbert`, `johnson2019faiss` |
| `sections/introduction.tex`, paragraph 2 | Reflection-loop context | `shinn2023reflexion`, `madaan2023selfrefine` |
| `sections/introduction.tex`, paragraph 3 | Standards-oriented export context | `dodaf2010`, `siso2020c2sim` |
| `sections/method.tex`, case-retrieval subsection | Retrieval and CBR framing | `lewis2020rag`, `reimers2019sbert`, `johnson2019faiss`, `aamodt1994cbr`, `kolodner1993cbr` |
| `sections/method.tex`, HOPE fast-weight subsection | Stochastic-update and nonlinear-control framing | `mao2007sde`, `khalil2002nonlinear` |
| `sections/method.tex`, reflection subsection | Reflection-loop framing | `shinn2023reflexion`, `madaan2023selfrefine` |
| `sections/method.tex`, simulation subsection | Monte Carlo simulation and operational-theory alignment framing | `jointchiefs2013jointtargeting`, `tirpak2000f2t2ea`, `alberts1999networkcentric`, `law2015simulation` |
| `sections/method.tex`, export subsection | DoDAF and C2SIM framing | `dodaf2010`, `siso2020c2sim` |

## Manuscript-to-Evidence Mapping

| Manuscript location | Evidence object | Primary source |
| --- | --- | --- |
| `sections/experimental_setup.tex`, ablation setup paragraph | Canonical ablation configuration | `result/ablation_suite_seed7_v3/ablation_*/full_result.json -> experiment_config` |
| `sections/experimental_setup.tex`, generalization setup paragraph | Canonical generalization configuration | `result/generalization_suite_seed7_v2_sceneout_iter20/<scene>/ablation_*/full_result.json -> experiment_config` |
| `sections/experimental_setup.tex`, runtime metadata paragraph | Refreshed model, environment, CPU, and GPU metadata for the main evidence bundles | `result/ablation_suite_seed7_v3/ablation_full/full_result.json -> runtime_manifest`; `result/generalization_suite_seed7_v2_sceneout_iter20/coastal_joint_assault/ablation_full/full_result.json -> runtime_manifest`; automatic capture implemented in `military_research/engine.py -> MilitaryResearchPipeline._collect_runtime_manifest(), run(), write_outputs()` |
| `sections/experimental_setup.tex`, reproducibility/reporting paragraph | Serialized wall-clock timing support for newly generated result bundles | `military_research/engine.py -> MilitaryResearchPipeline.run() -> runtime_stats`; `result/ablation_reflection_only_seed7/full_result.json -> runtime_stats` |
| `sections/results.tex`, Table `tab:ablation-main` | Single-scenario metrics by setting | `result/ablation_suite_seed7_v3/ablation_*/full_result.json -> best_simulation` |
| `sections/results.tex`, Table `tab:ablation-stage` | Single-scenario five-stage scores | `result/ablation_suite_seed7_v3/ablation_*/full_result.json -> best_simulation.stage_scores` |
| `sections/results.tex`, Figure `fig:ablation-stage-profile` | Single-scenario five-stage visualization across the four ablation settings | `result/ablation_suite_seed7_v3/ablation_*/full_result.json -> best_simulation.stage_scores`, rendered by `paper/manuscript/scripts/generate_paper_figures.py` |
| `sections/results.tex`, post-figure reflection paragraph | Reflection-only follow-up ablation under the same scenario and budget | `result/ablation_reflection_only_seed7/full_result.json -> best_simulation, runtime_stats` |
| `sections/results.tex`, Table `tab:generalization-summary` | Five-scene suite summary | `result/generalization_suite_seed7_v2_sceneout_iter20/generalization_summary.md` cross-checked against per-scene `full_result.json` |
| `sections/results.tex`, Table `tab:generalization-scenes` | Per-scene mission success and confidence intervals | `result/generalization_suite_seed7_v2_sceneout_iter20/<scene>/ablation_pure_llm/full_result.json -> best_simulation`; `.../ablation_full/full_result.json -> best_simulation, monte_carlo_stats.mission_success` |
| `sections/method.tex`, Figure `fig:method-overview` | Pipeline-level algorithm overview diagram | author-provided asset `paper/manuscript/figures/算法结构图.png`, copied to `paper/manuscript/figures/method_pipeline_architecture.png` for LaTeX inclusion |
| `sections/results.tex`, Figure `fig:generalization-bars` | Four-setting mission-success bars over five scenarios | `result/generalization_suite_seed7_v2_sceneout_iter20/<scene>/ablation_*/full_result.json -> best_simulation.mission_success`, rendered by `paper/manuscript/scripts/generate_paper_figures.py` |
| `sections/results.tex`, Figure `fig:full-trends` | Full-setting 50-iteration average trends over five scenarios | `result/generalization_suite_seed7_v2_sceneout_iter50/<scene>/ablation_full/full_result.json -> optimization_history[*].simulation.<metric>`, rendered by `paper/manuscript/scripts/generate_paper_figures.py` |
| `submission/graphical_abstract_notes.md` | Separate Graphical Abstract packaging note and asset set | `paper/manuscript/scripts/generate_graphical_abstract.py`; `paper/manuscript/figures/graphical_abstract_smpt.pdf`; `paper/manuscript/figures/graphical_abstract_smpt.png`; `paper/manuscript/figures/graphical_abstract_smpt.tiff` |

### Additional Table Evidence Added on 2026-05-18

| Manuscript location | Evidence object | Primary source |
| --- | --- | --- |
| `sections/experimental_setup.tex`, Table `tab:scenario-overview` | Five-scenario setup overview | `data/generalization_scenarios/*.json`; `data/generalization_case_banks_v2_sceneout/summary.json` |
| `sections/experimental_setup.tex`, Table `tab:metric-definitions` | Definitions of the five headline evaluation metrics | `military_research/engine.py` around mission-success, command-resilience, LER, completion-time, and overall-effectiveness computation; selected `full_result.json` field names |
| `sections/method.tex`, Table `tab:export-correspondence` | Representative mapping from structured plan fields to OV-5b, OV-6c, and C2SIM exports | `result/generalization_suite_seed7_v2_sceneout_iter20/coastal_joint_assault/ablation_full/best_plan.json`; `.../dodaf_ov5b.json`; `.../dodaf_ov6c.json`; `.../c2sim.xml` |

## Claim-to-Source Mapping

| Manuscript claim | Source | Notes |
| --- | --- | --- |
| The project is about Memento, Hope, Reflection, simulation evaluation, and DoDAF/C2SIM export rather than space-filling design. | `README.md` | High-level project description and run commands. |
| The domain layer defines capabilities, actions, scenarios, phases, and complete plans. | `military_research/domain.py` | See dataclasses such as `Scenario`, `PlanPhase`, and `CombatPlan`. |
| Case retrieval uses a FAISS-backed semantic path when available and falls back to keyword retrieval otherwise. | `military_research/case_memory.py` | `CaseBank.__init__`, `_load_backends`, `retrieve`. |
| Memory hits are re-ranked by support, quality, and confidence, then truncated to at most two in strict mode. | `military_research/engine.py` | `_select_memory_hits`, around lines 2152-2184. |
| The implemented memory path is `JSONL -> CaseBank load -> retrieval -> support-based reranking -> optional quality-gated write-back`. | `military_research/case_memory.py`; `military_research/engine.py` | `CaseRecord`, `_load_records`, `retrieve`, `_select_memory_hits`, `_write_back_case`, `upsert_record`. |
| Sparse scenes can receive synthetic prior case records for coastal assault, mountain recon, maritime sustainment, and urban defense. | `military_research/engine.py` | `_scene_memory_prior_records` and `_augment_memory_hits_with_priors`, around lines 2186-2369. |
| HOPEAdapter uses a value estimator, a forget gate with a log-age term, and a hidden state refreshed during feedback updates. | `military_research/engine.py` | `HOPEAdapter`, around lines 66-144. |
| Scenario encoding uses one-hot terrain/weather plus EW threat, threat level, and time pressure. | `military_research/engine.py` | `_encode_scenario`, around lines 197-225. |
| Fast weights are updated with an Euler-Maruyama-style step and clipped to scenario-dependent bounds. | `military_research/engine.py` | `fast_weights`, around lines 227-245. |
| Slow/fast capability fusion applies a scaled fast component and mission- or EW-specific floors. | `military_research/engine.py` | `fuse`, around lines 247-271. |
| Feedback integration computes stage deficits, adapter targets, a delay-correction term, and a bounded slow-weight update. | `military_research/engine.py` | `integrate_feedback`, around lines 294-397. |
| The simulator scores plans with five kill-chain stages and computes mission success, LER, survivability, command resilience, and overall effectiveness. | `military_research/engine.py` | `PlanSimulator.run`, around lines 1545-1694; `_evaluate_kill_chain`, around lines 1829-1862. |
| Monte Carlo summaries include mean, std, min, max, 95% CI, and n. | `military_research/engine.py` | `_aggregate_simulation_results`, around lines 2735-2795. |
| Refreshed canonical result bundles now include a top-level `runtime_manifest` with backend, software, and hardware metadata. | `military_research/engine.py`; `tests/test_runtime_manifest.py`; refreshed `full_result.json` files under `result/ablation_suite_seed7_v3/` and `result/generalization_suite_seed7_v2_sceneout_iter20/` | Older audit-only directories are not batch-rewritten. |
| Newly generated result bundles now also include top-level wall-clock timing summaries. | `military_research/engine.py`; `tests/test_runtime_manifest.py`; `result/ablation_reflection_only_seed7/full_result.json -> runtime_stats` | Captures total wall time, per-iteration wall time, candidate counts, and simulation rollouts. |
| The pipeline exports DoDAF OV-5b, OV-6c, and C2SIM artifacts. | `military_research/exporters.py` | `build_ov5b`, `build_ov6c`, `build_c2sim_xml`. |
| Export behavior is covered by tests. | `tests/test_exporters.py` | Checks structured actions and C2SIM XML typing. |
| Simulator Monte Carlo behavior and stage sensitivity are covered by tests. | `tests/test_simulator.py` | Checks Monte Carlo stats and action-type sensitivity. |
| Memory retrieval behavior and synthetic-prior backfill are covered by tests. | `tests/test_case_memory.py` | Checks negative retrieval, merge, usage mode, and prior backfill. |
| The final manuscript metadata names `deepseek-r1-0528-qwen3-8b` as the backend model, `Memento` as the Python environment, and specifies the CPU and GPU. | author-supplied confirmation dated 2026-05-16; refreshed runtime manifests dated 2026-05-23 | The main reported evidence is now backed by serialized `runtime_manifest` records rather than by author confirmation alone. |

## Quantitative Results Used in the Draft

### Single-scenario ablation (`result/ablation_suite_seed7_v3`)

All four settings are taken from the corresponding `full_result.json` files under:

- `ablation_pure_llm/`
- `ablation_memento/`
- `ablation_hope/`
- `ablation_full/`

Key experiment configuration extracted from `experiment_config`:

- seed = 7
- iterations = 10
- sim_runs = 50
- allow_writeback = `false` for all four settings
- scenario path = `data/sample_joint_operation.json`

Main values copied into the manuscript:

| Setting | Mission success | Overall effectiveness | LER | Time (h) | Command resilience |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure LLM | 0.6260 | 0.7328 | 1.6564 | 11.41 | 0.6096 |
| LLM + Memento | 0.6271 | 0.7221 | 1.6301 | 14.59 | 0.6096 |
| LLM + Hope | 0.6229 | 0.7572 | 1.7550 | 9.93 | 0.7156 |
| Full | 0.6388 | 0.7685 | 1.9771 | 14.14 | 0.6852 |

Line-level provenance used for the manuscript tables:

- `Pure LLM` row: `result/ablation_suite_seed7_v3/ablation_pure_llm/full_result.json -> best_simulation`
- `LLM + Memento` row: `result/ablation_suite_seed7_v3/ablation_memento/full_result.json -> best_simulation`
- `LLM + Hope` row: `result/ablation_suite_seed7_v3/ablation_hope/full_result.json -> best_simulation`
- `Full` row: `result/ablation_suite_seed7_v3/ablation_full/full_result.json -> best_simulation`

Reflection-only rerun now integrated into the main single-scenario result view:

| Setting | Mission success | Overall effectiveness | LER | Time (h) | Command resilience |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reflection only | 0.6340 | 0.7447 | 1.7498 | 11.39 | 0.6096 |

Additional reflection-only stage scores:

- detect = 0.6281
- disrupt = 0.5961
- breach = 0.5816
- control = 0.6344
- sustain = 0.6548

Reflection-only provenance:

- `result/ablation_reflection_only_seed7/full_result.json -> best_simulation`
- `result/ablation_reflection_only_seed7/full_result.json -> runtime_stats.total_wall_time_sec = 327.3831`

Second-round integration note:

- Reflection-only is now part of the evidence chain for `Table 4`, `Table 5`, and `Figure 2`, rather than remaining only as a post-table narrative note.
- `Figure 3` now adds panel-level `y-range` annotations while preserving the same underlying mission-success values and scene ordering.
- The revised abstract and discussion explicitly treat the reported confidence intervals as within-seed Monte Carlo variance rather than across-seed variance.
- The five-scene transfer suite remains a four-setting comparison and does not add a reflection-only arm.

Full-stage scores copied into the manuscript:

- detect = 0.6303
- disrupt = 0.6038
- breach = 0.5890
- control = 0.6590
- sustain = 0.6874

### Cross-scenario generalization (`result/generalization_suite_seed7_v2_sceneout_iter20`)

Suite-level summary source:

- `result/generalization_suite_seed7_v2_sceneout_iter20/generalization_summary.md`

Scene-level primary source:

- `result/generalization_suite_seed7_v2_sceneout_iter20/<scene>/ablation_*/full_result.json`

Key suite properties:

- scene count = 5
- iterations = 20 per scene
- sim_runs = 50 per scene
- seed = 7
- case-bank summary source = `data/generalization_case_banks_v2_sceneout/summary.json`
- source case count = 250
- scene-out case count = 200 for each of the five scenario banks

Values copied into the manuscript:

- Average mission success: Pure = 0.6595, Full = 0.6828
- Average overall effectiveness: Pure = 0.7417, Full = 0.7924
- Full beats Pure LLM in 5/5 scenarios
- Full is not lower than Pure/Memento/Hope in 4/5 scenarios

Scene-level values copied into the manuscript:

| Scene | Pure MS | Full MS | Delta | 95% CI of Full MS |
| --- | ---: | ---: | ---: | --- |
| Coastal joint assault | 0.6332 | 0.6847 | +0.0515 | [0.6840, 0.6855] |
| Island resupply corridor | 0.6405 | 0.6643 | +0.0238 | [0.6636, 0.6649] |
| Mountain corridor reconnaissance | 0.6859 | 0.7045 | +0.0186 | [0.7035, 0.7054] |
| River crossing breakthrough | 0.6573 | 0.6698 | +0.0125 | [0.6691, 0.6705] |
| Urban hub defense | 0.6805 | 0.6909 | +0.0104 | [0.6901, 0.6918] |

Line-level provenance used for the manuscript scene table:

- Coastal assault row:
  - baseline: `result/generalization_suite_seed7_v2_sceneout_iter20/coastal_joint_assault/ablation_pure_llm/full_result.json -> best_simulation.mission_success`
  - full: `result/generalization_suite_seed7_v2_sceneout_iter20/coastal_joint_assault/ablation_full/full_result.json -> best_simulation.mission_success, monte_carlo_stats.mission_success`
- Island resupply row:
  - baseline: `result/generalization_suite_seed7_v2_sceneout_iter20/island_resupply_corridor/ablation_pure_llm/full_result.json -> best_simulation.mission_success`
  - full: `result/generalization_suite_seed7_v2_sceneout_iter20/island_resupply_corridor/ablation_full/full_result.json -> best_simulation.mission_success, monte_carlo_stats.mission_success`
- Mountain reconnaissance row:
  - baseline: `result/generalization_suite_seed7_v2_sceneout_iter20/mountain_corridor_recon/ablation_pure_llm/full_result.json -> best_simulation.mission_success`
  - full: `result/generalization_suite_seed7_v2_sceneout_iter20/mountain_corridor_recon/ablation_full/full_result.json -> best_simulation.mission_success, monte_carlo_stats.mission_success`
- River crossing row:
  - baseline: `result/generalization_suite_seed7_v2_sceneout_iter20/river_crossing_breakthrough/ablation_pure_llm/full_result.json -> best_simulation.mission_success`
  - full: `result/generalization_suite_seed7_v2_sceneout_iter20/river_crossing_breakthrough/ablation_full/full_result.json -> best_simulation.mission_success, monte_carlo_stats.mission_success`
- Urban defense row:
  - baseline: `result/generalization_suite_seed7_v2_sceneout_iter20/urban_hub_defense/ablation_pure_llm/full_result.json -> best_simulation.mission_success`
  - full: `result/generalization_suite_seed7_v2_sceneout_iter20/urban_hub_defense/ablation_full/full_result.json -> best_simulation.mission_success, monte_carlo_stats.mission_success`

## Figure-Derivation Notes

- `generalization_scene_group_bar.png` is derived from the canonical 20-iteration scene-out suite and plots `best_simulation.mission_success` for the four settings (`ablation_pure_llm`, `ablation_memento`, `ablation_hope`, `ablation_full`) in five scenario-specific panels, one panel per selected scenario, with locally tightened y-axis ranges.
- `full_iter50_average_trends.png` is derived from the auxiliary 50-iteration scene-out suite and plots iteration-wise arithmetic means across the five `ablation_full/full_result.json` optimization histories for:
  - `mission_success`
  - `overall_effectiveness`
  - `ler`
  - `completion_time_hours`
  - `command_resilience`
- For the trend figure, the published panel lines use both the raw iteration-wise means and a centered five-iteration moving average for visual smoothing.
- The figure-generation script also exports CSV companions under `paper/manuscript/figures/` so that the plotted values can be inspected without re-reading the raw JSON files.

## Build Notes

- A workspace-local TinyTeX distribution is installed under `.tools/tinytex/dist/TinyTeX`.
- `elsarticle.cls` was generated from `paper/template_elsarticle/elsarticle/elsarticle.ins`.
- `paper/manuscript/main.tex` compiled successfully to `paper/manuscript/main.pdf` on 2026-05-16 using `latexmk`, `pdflatex`, and `bibtex`.

## Excluded Materials

- `Figure/f1_ood.jpg`, `Figure/f1_iteration.jpg`, `Figure/f1_tasks.jpg`, and `Figure/f1_val_test.jpg` were inspected visually and appear unrelated to the current planning-and-simulation evidence line. They are excluded from the manuscript.
- Other result directories under `result/` were not used in the main body in order to avoid mixing evidence sets with different case-bank constructions or iteration budgets.

## Known Consistency Notes

- The ablation summary markdown is useful for quick reading, but the manuscript should trust the raw `full_result.json` files for final experiment configuration fields.
- In particular, the selected evidence records `allow_writeback = false`, so the manuscript writes the experiment setup accordingly.
- The repository README contains an example local environment variable `LOCAL_LLM_MODEL=openai/gpt-oss-20b`, but the final manuscript metadata and refreshed rerun manifests rely on `deepseek-r1-0528-qwen3-8b` rather than that README example.
- Figure `fig:method-overview` was checked visually against `paper/manuscript/figures/method_pipeline_architecture.png` on 2026-05-23. The earlier apparent typos reported from PDF/OCR extraction were not present in the source asset and are treated as text-layer/OCR noise rather than as genuine artwork spelling errors.
