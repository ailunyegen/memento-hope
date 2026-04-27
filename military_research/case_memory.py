from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CaseRecord":
        return cls(**payload)

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

        self._SentenceTransformer = None
        self._faiss = None
        self._encoder = None

        self._load_records()
        self._load_backends()
        self._rebuild_index()

    def add_record(self, record: CaseRecord) -> None:
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._rebuild_index()

    def retrieve(self, scenario: Scenario, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.records or self.index is None or self.dimension == 0:
            return []

        query_embedding = self._encode_texts([self._scenario_to_text(scenario)])
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.records)))

        ranked: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            confidence = max(0.0, min(1.0, float(score)))
            ranked.append(
                {
                    "score": round(float(score), 4),
                    "confidence": round(confidence, 4),
                    "record": self.records[int(index)],
                }
            )
        return ranked

    def _load_records(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(CaseRecord.from_dict(json.loads(line)))

    def _load_backends(self) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "FAISS memory requires `sentence-transformers` and `faiss-cpu`. "
                "Please install project dependencies before running the research pipeline."
            ) from exc

        self._faiss = faiss
        self._SentenceTransformer = SentenceTransformer

    def _get_encoder(self):
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
            return

        texts = [self._record_to_text(record) for record in self.records]
        embeddings = self._encode_texts(texts)
        self.embedding_matrix = embeddings
        self.dimension = int(embeddings.shape[1])

        assert self._faiss is not None
        index = self._faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        self.index = index

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
