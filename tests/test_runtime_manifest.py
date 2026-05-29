from __future__ import annotations

from military_research.domain import ForceUnit, Scenario
from military_research.engine import MilitaryResearchPipeline


def make_runtime_test_scenario() -> Scenario:
    return Scenario(
        name="runtime-test",
        mission_type="assault",
        objective="Validate runtime reporting",
        terrain="coastal-urban",
        weather="rain",
        threat_level=0.72,
        ew_threat=0.65,
        civilian_presence=0.2,
        time_pressure=0.5,
        doctrine_profile="offensive_breakthrough",
        friendly_forces=[
            ForceUnit(name="Recon Unit", role="recon", domain="air", awareness=0.9, c2=0.7, ew=0.7),
            ForceUnit(name="Fire Unit", role="fires", domain="land", fires=0.9),
            ForceUnit(name="Assault Unit", role="assault", domain="land", mobility=0.8, protection=0.7),
            ForceUnit(name="Support Unit", role="sustain", domain="land", sustainment=0.8),
        ],
        enemy_forces=[ForceUnit(name="Defense Group", role="defense", domain="land")],
    )


def test_runtime_manifest_uses_current_backend_metadata(monkeypatch, tmp_path) -> None:
    case_bank_path = tmp_path / "cases.jsonl"
    case_bank_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "deepseek-r1-0528-qwen3-8b")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("RUNTIME_CPU_NAME", "13th Gen Intel(R) Core(TM) i7-13700F")
    monkeypatch.setenv("RUNTIME_GPU_NAME", "NVIDIA GeForce RTX 3060")

    pipeline = MilitaryResearchPipeline(case_bank_path)
    manifest = pipeline._collect_runtime_manifest()

    assert set(manifest) == {"collected_at", "llm_backend", "software", "hardware"}
    assert manifest["llm_backend"]["model_id"] == "deepseek-r1-0528-qwen3-8b"
    assert manifest["llm_backend"]["base_url"] == "http://localhost:1234/v1"
    assert manifest["llm_backend"]["api_mode"] == "openai-compatible"
    assert manifest["software"]["python_version"]
    assert "api_key" not in str(manifest).lower()
    assert manifest["hardware"]["cpu"] == "13th Gen Intel(R) Core(TM) i7-13700F"
    assert manifest["hardware"]["gpu"] == "NVIDIA GeForce RTX 3060"
    assert isinstance(manifest["hardware"]["cuda_available"], bool)


def test_runtime_manifest_marks_unknown_model_when_unset(monkeypatch, tmp_path) -> None:
    case_bank_path = tmp_path / "cases.jsonl"
    case_bank_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)

    pipeline = MilitaryResearchPipeline(case_bank_path)
    manifest = pipeline._collect_runtime_manifest()

    assert manifest["llm_backend"]["model_id"] == "unknown"
    assert manifest["llm_backend"]["base_url"] == "http://localhost:1234/v1"


def test_pipeline_run_serializes_runtime_stats(tmp_path) -> None:
    case_bank_path = tmp_path / "cases.jsonl"
    case_bank_path.write_text("", encoding="utf-8")

    pipeline = MilitaryResearchPipeline(case_bank_path, seed=7)
    result = pipeline.run(
        make_runtime_test_scenario(),
        iterations=1,
        disable_memory=True,
        disable_hope=True,
        disable_reflection=True,
        sim_runs=2,
        allow_writeback=False,
    )

    runtime_stats = result["runtime_stats"]
    assert runtime_stats["total_wall_time_sec"] >= 0.0
    assert runtime_stats["avg_iteration_wall_time_sec"] >= 0.0
    assert runtime_stats["iteration_wall_time_sec"]
    assert runtime_stats["total_candidate_evaluations"] >= 1
    assert runtime_stats["avg_candidates_per_iteration"] >= 1.0
    assert runtime_stats["total_simulation_rollouts"] == 2
    assert runtime_stats["sim_runs_per_evaluation"] == 2
