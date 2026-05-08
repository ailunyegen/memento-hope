from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .domain import Scenario


@dataclass
class CaseRecord:
    case_id: str
    mission_type: str
    terrain: str
    objective: str
    friendly_roles: List[str]
    enemy_roles: List[str]
    key_actions: List[str]
    lessons: List[str]
    outcome: str
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    update_count: int = 1
    last_updated: str = ""
    source_scenarios: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CaseRecord":
        data = dict(payload)
        data.setdefault("update_count", 1)
        data.setdefault("last_updated", "")
        data.setdefault("source_scenarios", [])
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CaseBank:
    def __init__(
        self,
        path: str | Path,
        embedding_model: str | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model or os.getenv(
            "CASE_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.local_files_only = os.getenv("CASE_EMBEDDING_LOCAL_ONLY", "0") == "1"
        self.records: List[CaseRecord] = []
        self.index = None
        self.embedding_matrix = np.zeros((0, 0), dtype=np.float32)
        self.dimension = 0
        self.backend = "keyword"
        self._record_texts: List[str] = []

        self._SentenceTransformer = None
        self._faiss = None
        self._encoder = None

        self._load_records()
        self._load_backends()
        try:
            self._rebuild_index()
        except Exception:
            self.backend = "keyword"
            self.index = None
            self.embedding_matrix = np.zeros((0, 0), dtype=np.float32)
            self.dimension = 0
            self._encoder = None
            self._rebuild_index()

    def add_record(self, record: CaseRecord) -> None:
        self._append_record(record, persist_mode="append")

    def upsert_record(self, record: CaseRecord, similarity_threshold: float = 0.94) -> Tuple[str, CaseRecord]:
        similar_index, _ = self.find_similar_record(record, similarity_threshold=similarity_threshold)
        if similar_index is None:
            self._append_record(record, persist_mode="append")
            return "inserted", record

        merged_record = self._merge_records(self.records[similar_index], record)
        self._replace_record(similar_index, merged_record)
        return "updated", merged_record

    def retrieve(self, scenario: Scenario, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.records:
            return []
        if self.backend == "keyword":
            return self._keyword_retrieve(scenario, top_k=top_k)
        if self.index is None or self.dimension == 0:
            return []

        query_embedding = self._encode_texts([self._scenario_to_text(scenario)])
        search_k = min(max(top_k * 4, top_k), len(self.records))
        scores, indices = self.index.search(query_embedding, search_k)

        ranked: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = self.records[int(index)]
            quality = self._quality_score(record)
            if not self._eligible_for_retrieval(record, quality):
                continue
            confidence = max(0.0, min(1.0, float(score)))
            blended_score = 0.72 * float(score) + 0.28 * quality
            ranked.append(
                {
                    "score": round(float(score), 4),
                    "blended_score": round(float(blended_score), 4),
                    "confidence": round(confidence, 4),
                    "quality": round(float(quality), 4),
                    "usage_mode": "imitate" if record.outcome == "positive" else "avoid",
                    "record": record,
                }
            )
        ranked.sort(key=lambda item: (item["blended_score"], item["confidence"]), reverse=True)
        return ranked[:top_k]

    def find_similar_record(
        self,
        record: CaseRecord,
        similarity_threshold: float = 0.94,
    ) -> Tuple[int | None, float]:
        if not self.records:
            return None, 0.0
        if self.backend == "keyword":
            return self._keyword_find_similar_record(record, similarity_threshold)
        if self.index is None or self.dimension == 0:
            return None, 0.0

        record_text = self._record_to_text(record)
        query_embedding = self._encode_texts([record_text])
        scores, indices = self.index.search(query_embedding, 1)
        if len(scores[0]) == 0 or len(indices[0]) == 0 or indices[0][0] < 0:
            return None, 0.0

        candidate_index = int(indices[0][0])
        score = float(scores[0][0])
        candidate = self.records[candidate_index]
        if score < similarity_threshold:
            return None, score
        if not self._structurally_compatible(candidate, record):
            return None, score
        return candidate_index, score

    def _load_records(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = CaseRecord.from_dict(json.loads(line))
                    self.records.append(record)
                    self._record_texts.append(self._record_to_text(record))

    def _load_backends(self) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._faiss = None
            self._SentenceTransformer = None
            self.backend = "keyword"
            return

        self._faiss = faiss
        self._SentenceTransformer = SentenceTransformer
        self.backend = "faiss"

    def _get_encoder(self):
        if self.backend != "faiss":
            raise RuntimeError("Encoder requested while CaseBank is using keyword backend.")
        if self._encoder is None:
            assert self._SentenceTransformer is not None
            self._encoder = self._SentenceTransformer(
                self.embedding_model_name,
                local_files_only=self.local_files_only,
            )
        return self._encoder

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _rebuild_index(self) -> None:
        if not self.records:
            self.index = None
            self.embedding_matrix = np.zeros((0, 0), dtype=np.float32)
            self.dimension = 0
            self._record_texts = []
            return

        self._record_texts = [self._record_to_text(record) for record in self.records]
        if self.backend != "faiss":
            self.index = True
            self.embedding_matrix = np.zeros((0, 0), dtype=np.float32)
            self.dimension = 1
            return
        embeddings = self._encode_texts(self._record_texts)
        self.embedding_matrix = embeddings
        self.dimension = int(embeddings.shape[1])
        self._rebuild_index_from_embeddings()

    def _append_record(self, record: CaseRecord, persist_mode: str = "append") -> None:
        record_text = self._record_to_text(record)

        self.records.append(record)
        self._record_texts.append(record_text)
        if self.backend == "faiss":
            embedding = self._encode_texts([record_text])
            if self.dimension == 0:
                self.embedding_matrix = embedding
                self.dimension = int(embedding.shape[1])
                self._rebuild_index_from_embeddings()
            else:
                self.embedding_matrix = np.vstack([self.embedding_matrix, embedding])
                assert self.index is not None
                self.index.add(embedding)
        else:
            self.index = True
            self.dimension = 1

        if persist_mode == "append":
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        else:
            self._persist_records()

    def _replace_record(self, index: int, record: CaseRecord) -> None:
        record_text = self._record_to_text(record)
        self.records[index] = record
        self._record_texts[index] = record_text
        if self.backend == "faiss":
            embedding = self._encode_texts([record_text])[0]
            self.embedding_matrix[index] = embedding
            self.dimension = int(self.embedding_matrix.shape[1])
            self._rebuild_index_from_embeddings()
        else:
            self.index = True
            self.dimension = 1
        self._persist_records()

    def _persist_records(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def _rebuild_index_from_embeddings(self) -> None:
        if self.backend != "faiss":
            self.index = True
            return
        if self.dimension == 0 or self.embedding_matrix.size == 0:
            self.index = None
            return

        assert self._faiss is not None
        index = self._faiss.IndexFlatIP(self.dimension)
        index.add(self.embedding_matrix)
        self.index = index

    def _keyword_retrieve(self, scenario: Scenario, top_k: int = 3) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for record in self.records:
            quality = self._quality_score(record)
            if not self._eligible_for_retrieval(record, quality):
                continue
            confidence = self._keyword_similarity(self._scenario_keyword_features(scenario), self._record_keyword_features(record))
            blended_score = 0.68 * confidence + 0.32 * quality
            ranked.append(
                {
                    "score": round(confidence, 4),
                    "blended_score": round(blended_score, 4),
                    "confidence": round(confidence, 4),
                    "quality": round(float(quality), 4),
                    "usage_mode": "imitate" if record.outcome == "positive" else "avoid",
                    "record": record,
                }
            )
        ranked.sort(key=lambda item: (item["blended_score"], item["confidence"]), reverse=True)
        return ranked[:top_k]

    def _keyword_find_similar_record(
        self,
        record: CaseRecord,
        similarity_threshold: float = 0.94,
    ) -> Tuple[int | None, float]:
        best_index: int | None = None
        best_score = 0.0
        target = self._record_keyword_features(record)
        for index, candidate in enumerate(self.records):
            if not self._structurally_compatible(candidate, record):
                continue
            score = self._keyword_similarity(target, self._record_keyword_features(candidate))
            if score > best_score:
                best_score = score
                best_index = index
        adaptive_threshold = min(similarity_threshold, 0.78)
        if best_index is None or best_score < adaptive_threshold:
            return None, best_score
        return best_index, best_score

    def _scenario_keyword_features(self, scenario: Scenario) -> Dict[str, Any]:
        return {
            "mission_type": self._normalize_text(scenario.mission_type),
            "terrain": self._normalize_text(scenario.terrain),
            "objective": self._normalize_text(scenario.objective),
            "friendly_roles": {self._normalize_text(unit.role) for unit in scenario.friendly_forces},
            "enemy_roles": {self._normalize_text(unit.role) for unit in scenario.enemy_forces},
            "tags": {
                self._normalize_text(scenario.weather),
                self._normalize_text(scenario.doctrine_profile),
                self._normalize_text(scenario.name),
            },
            "actions": set(),
        }

    def _record_keyword_features(self, record: CaseRecord) -> Dict[str, Any]:
        return {
            "mission_type": self._normalize_text(record.mission_type),
            "terrain": self._normalize_text(record.terrain),
            "objective": self._normalize_text(record.objective),
            "friendly_roles": {self._normalize_text(role) for role in record.friendly_roles},
            "enemy_roles": {self._normalize_text(role) for role in record.enemy_roles},
            "tags": {self._normalize_text(tag) for tag in record.tags},
            "actions": {self._normalize_text(action) for action in record.key_actions},
        }

    def _keyword_similarity(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        score = 0.0
        weights = {
            "mission_type": 0.22,
            "terrain": 0.18,
            "objective": 0.18,
            "friendly_roles": 0.12,
            "enemy_roles": 0.12,
            "tags": 0.08,
            "actions": 0.10,
        }
        for key, weight in weights.items():
            left_value = left.get(key)
            right_value = right.get(key)
            if isinstance(left_value, set) and isinstance(right_value, set):
                union = left_value | right_value
                overlap = 0.0 if not union else len(left_value & right_value) / len(union)
                score += weight * overlap
            elif left_value and right_value and left_value == right_value:
                score += weight
        return max(0.0, min(1.0, score))

    def _merge_records(self, existing: CaseRecord, incoming: CaseRecord) -> CaseRecord:
        merged_metrics = dict(existing.metrics)
        for key, value in incoming.metrics.items():
            previous = merged_metrics.get(key)
            if previous is None:
                merged_metrics[key] = value
            elif key in {"mission_success", "ler", "overall_effectiveness"}:
                merged_metrics[key] = max(previous, value)
            elif key.startswith("stage_"):
                merged_metrics[key] = max(previous, value)
            else:
                merged_metrics[key] = value

        merged_actions = self._dedupe_text_items(existing.key_actions + incoming.key_actions, limit=8)
        merged_lessons = self._dedupe_text_items(existing.lessons + incoming.lessons, limit=6)
        merged_tags = self._dedupe_text_items(existing.tags + incoming.tags, limit=8)
        merged_scenarios = self._dedupe_text_items(existing.source_scenarios + incoming.source_scenarios, limit=12)
        merged_outcome = "positive" if "positive" in {existing.outcome, incoming.outcome} else incoming.outcome

        return CaseRecord(
            case_id=existing.case_id,
            mission_type=existing.mission_type,
            terrain=existing.terrain,
            objective=existing.objective,
            friendly_roles=self._dedupe_text_items(existing.friendly_roles + incoming.friendly_roles),
            enemy_roles=self._dedupe_text_items(existing.enemy_roles + incoming.enemy_roles),
            key_actions=merged_actions,
            lessons=merged_lessons,
            outcome=merged_outcome,
            tags=merged_tags,
            metrics=merged_metrics,
            update_count=max(existing.update_count, 1) + 1,
            last_updated=incoming.last_updated or existing.last_updated or self._now_timestamp(),
            source_scenarios=merged_scenarios,
        )

    def _structurally_compatible(self, candidate: CaseRecord, incoming: CaseRecord) -> bool:
        if self._normalize_text(candidate.mission_type) != self._normalize_text(incoming.mission_type):
            return False
        if self._normalize_text(candidate.terrain) != self._normalize_text(incoming.terrain):
            return False

        candidate_objective = self._normalize_text(candidate.objective)
        incoming_objective = self._normalize_text(incoming.objective)
        if candidate_objective != incoming_objective and not (
            candidate_objective in incoming_objective or incoming_objective in candidate_objective
        ):
            return False

        candidate_friendly = set(self._normalize_text(role) for role in candidate.friendly_roles)
        incoming_friendly = set(self._normalize_text(role) for role in incoming.friendly_roles)
        candidate_enemy = set(self._normalize_text(role) for role in candidate.enemy_roles)
        incoming_enemy = set(self._normalize_text(role) for role in incoming.enemy_roles)

        return candidate_friendly == incoming_friendly and candidate_enemy == incoming_enemy

    def _dedupe_text_items(self, items: List[str], limit: int | None = None) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for item in items:
            normalized = self._normalize_text(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(item)
            if limit is not None and len(deduped) >= limit:
                break
        return deduped

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    def _now_timestamp(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _quality_score(self, record: CaseRecord) -> float:
        metrics = record.metrics or {}
        mission_success = float(metrics.get("mission_success", 0.0))
        overall = float(metrics.get("overall_effectiveness", 0.0))
        ler = float(metrics.get("ler", 1.0))
        command_resilience = float(metrics.get("command_resilience", 0.58))
        stage_scores = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("stage_")
        ]
        stage_quality = min(stage_scores) if stage_scores else 0.55
        update_bonus = min(max(record.update_count, 1), 4) * 0.015
        auto_penalty = 0.04 if record.case_id.startswith("AUTO-") else 0.0
        negative_penalty = 0.12 if record.outcome != "positive" else 0.0
        ler_score = min(ler / 2.2, 1.0)
        quality = (
            0.38 * mission_success
            + 0.24 * overall
            + 0.16 * stage_quality
            + 0.12 * ler_score
            + 0.10 * command_resilience
            + update_bonus
            - auto_penalty
            - negative_penalty
        )
        return float(max(0.0, min(1.0, quality)))

    def _eligible_for_retrieval(self, record: CaseRecord, quality: float) -> bool:
        metrics = record.metrics or {}
        mission_success = float(metrics.get("mission_success", 0.0))
        overall = float(metrics.get("overall_effectiveness", 0.0))
        stage_scores = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("stage_")
        ]
        min_stage = min(stage_scores) if stage_scores else 0.0

        if record.outcome == "positive" and (mission_success >= 0.60 or overall >= 0.72):
            return True
        if record.outcome == "positive" and quality >= 0.60 and min_stage >= 0.50:
            return True
        if record.outcome != "positive" and quality >= 0.28:
            return True
        return False

    def _record_to_text(self, record: CaseRecord) -> str:
        return "\n".join(
            [
                f"mission_type: {record.mission_type}",
                f"terrain: {record.terrain}",
                f"objective: {record.objective}",
                f"friendly_roles: {'; '.join(record.friendly_roles)}",
                f"enemy_roles: {'; '.join(record.enemy_roles)}",
                f"key_actions: {'; '.join(record.key_actions)}",
                f"lessons: {'; '.join(record.lessons)}",
                f"outcome: {record.outcome}",
                f"tags: {'; '.join(record.tags)}",
                f"source_scenarios: {'; '.join(record.source_scenarios)}",
            ]
        )

    def _scenario_to_text(self, scenario: Scenario) -> str:
        return "\n".join(
            [
                f"name: {scenario.name}",
                f"mission_type: {scenario.mission_type}",
                f"objective: {scenario.objective}",
                f"terrain: {scenario.terrain}",
                f"weather: {scenario.weather}",
                f"threat_level: {scenario.threat_level}",
                f"ew_threat: {scenario.ew_threat}",
                f"civilian_presence: {scenario.civilian_presence}",
                f"time_pressure: {scenario.time_pressure}",
                f"doctrine_profile: {scenario.doctrine_profile}",
                f"constraints: {'; '.join(scenario.constraints)}",
                f"friendly_forces: {'; '.join(f'{unit.name}/{unit.role}/{unit.domain}' for unit in scenario.friendly_forces)}",
                f"enemy_forces: {'; '.join(f'{unit.name}/{unit.role}/{unit.domain}' for unit in scenario.enemy_forces)}",
            ]
        )
