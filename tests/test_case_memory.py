from __future__ import annotations

from military_research.case_memory import CaseBank, CaseRecord
from military_research.domain import ForceUnit, Scenario
from military_research.engine import MilitaryResearchPipeline


def test_negative_case_can_be_retrieved_as_avoid_sample() -> None:
    bank = CaseBank.__new__(CaseBank)
    negative = CaseRecord(
        case_id="CASE-NEG",
        mission_type="assault",
        terrain="coastal",
        objective="夺控窗口",
        friendly_roles=["assault"],
        enemy_roles=["defense"],
        key_actions=["直接突击"],
        lessons=["未先侦察导致损失扩大"],
        outcome="negative",
        metrics={"mission_success": 0.35, "overall_effectiveness": 0.33, "ler": 0.7},
    )
    quality = bank._quality_score(negative)
    assert bank._eligible_for_retrieval(negative, quality) is True


def test_merge_record_keeps_better_metrics() -> None:
    bank = CaseBank.__new__(CaseBank)
    existing = CaseRecord(
        case_id="CASE-001",
        mission_type="assault",
        terrain="coastal",
        objective="obj",
        friendly_roles=["A"],
        enemy_roles=["B"],
        key_actions=["侦察"],
        lessons=["lesson-1"],
        outcome="positive",
        metrics={"mission_success": 0.55, "overall_effectiveness": 0.60},
    )
    incoming = CaseRecord(
        case_id="AUTO-1",
        mission_type="assault",
        terrain="coastal",
        objective="obj",
        friendly_roles=["A"],
        enemy_roles=["B"],
        key_actions=["压制"],
        lessons=["lesson-2"],
        outcome="positive",
        metrics={"mission_success": 0.64, "overall_effectiveness": 0.72},
    )
    merged = bank._merge_records(existing, incoming)
    assert merged.metrics["mission_success"] == 0.64
    assert merged.metrics["overall_effectiveness"] == 0.72
    assert merged.update_count == 2


def test_keyword_backend_retrieval_keeps_usage_mode() -> None:
    bank = CaseBank.__new__(CaseBank)
    bank.backend = "keyword"
    bank.index = True
    bank.dimension = 1
    bank.records = [
        CaseRecord(
            case_id="CASE-POS",
            mission_type="assault",
            terrain="coastal",
            objective="夺控桥头堡",
            friendly_roles=["recon", "assault"],
            enemy_roles=["defense"],
            key_actions=["侦察塑形", "火力压制"],
            lessons=["先侦察后压制"],
            outcome="positive",
            tags=["rain", "offensive_breakthrough"],
            metrics={"mission_success": 0.72, "overall_effectiveness": 0.75, "ler": 1.4},
        ),
        CaseRecord(
            case_id="CASE-NEG",
            mission_type="assault",
            terrain="coastal",
            objective="夺控桥头堡",
            friendly_roles=["assault"],
            enemy_roles=["defense"],
            key_actions=["直接突击"],
            lessons=["未压制导致失败"],
            outcome="negative",
            tags=["rain"],
            metrics={"mission_success": 0.35, "overall_effectiveness": 0.32, "ler": 0.7},
        ),
    ]
    scenario = Scenario(
        name="demo",
        mission_type="assault",
        objective="夺控桥头堡",
        terrain="coastal",
        weather="rain",
        threat_level=0.6,
        ew_threat=0.5,
        civilian_presence=0.2,
        time_pressure=0.4,
        doctrine_profile="offensive_breakthrough",
        friendly_forces=[ForceUnit(name="侦察分队", role="recon", domain="air")],
        enemy_forces=[ForceUnit(name="敌防御群", role="defense", domain="land")],
    )
    hits = bank.retrieve(scenario, top_k=2)
    assert hits
    assert {hit["usage_mode"] for hit in hits}.issubset({"imitate", "avoid"})


def test_memory_priors_backfill_sparse_generalization_scene() -> None:
    pipeline = MilitaryResearchPipeline.__new__(MilitaryResearchPipeline)
    scenario = Scenario(
        name="island-demo",
        mission_type="sustain",
        objective="保持岛链补给走廊畅通",
        terrain="maritime-island",
        weather="windy",
        threat_level=0.6,
        ew_threat=0.5,
        civilian_presence=0.1,
        time_pressure=0.4,
        doctrine_profile="balanced_joint",
        friendly_forces=[
            ForceUnit(name="补给群", role="海上补给", domain="sea"),
            ForceUnit(name="护航单元", role="护航防御", domain="sea"),
            ForceUnit(name="通信单元", role="指挥通信", domain="information"),
        ],
        enemy_forces=[ForceUnit(name="敌侦打群", role="远程侦打", domain="air")],
    )
    augmented = pipeline._augment_memory_hits_with_priors(scenario, [])
    selected = pipeline._select_memory_hits(scenario, augmented, strict=True)
    assert selected
    assert selected[0]["record"].case_id == "PRIOR-MARITIME-SUSTAIN"


def test_urban_prior_can_backfill_sparse_defense_scene() -> None:
    pipeline = MilitaryResearchPipeline.__new__(MilitaryResearchPipeline)
    scenario = Scenario(
        name="urban-demo",
        mission_type="defense",
        objective="稳固城市交通枢纽",
        terrain="urban",
        weather="cloudy",
        threat_level=0.6,
        ew_threat=0.4,
        civilian_presence=0.4,
        time_pressure=0.5,
        doctrine_profile="defense_in_depth",
        friendly_forces=[
            ForceUnit(name="机械步兵营", role="机动防御", domain="ground"),
            ForceUnit(name="通信分队", role="指挥通信", domain="information"),
        ],
        enemy_forces=[ForceUnit(name="敌装甲群", role="装甲突击", domain="ground")],
    )
    augmented = pipeline._augment_memory_hits_with_priors(scenario, [])
    selected = pipeline._select_memory_hits(scenario, augmented, strict=True)
    assert selected
    assert selected[0]["record"].case_id == "PRIOR-URBAN-DEFENSE"
