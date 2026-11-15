"""run_config - the single switch that separates the two arms.

Everything the comparison holds constant (model, prompt, index, embedding
model, scenarios, judge) is *not* in here. Only the four things under study are.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Literal

from src.config import RETRIEVAL_TOP_K, TOTAL_BUDGET


@dataclass(frozen=True)
class RunConfig:
    arm: Literal["naive", "engineered"]
    routing: bool
    memory: bool
    enforce_budget: bool
    retrieval_k: int = RETRIEVAL_TOP_K
    total_budget: int | None = TOTAL_BUDGET
    max_history_tokens: int | None = None
    pipeline_version: str = "context-v3"

    @classmethod
    def naive(cls, *, retrieval_k: int = RETRIEVAL_TOP_K) -> "RunConfig":
        """Stock RAG: always retrieve, no routing, no memory, no budget.

        ``total_budget=None`` keeps the enforcer node in the graph as a
        pass-through so the record still shows the tokens it would have saved.
        """
        return cls(
            arm="naive",
            routing=False,
            memory=False,
            enforce_budget=False,
            retrieval_k=retrieval_k,
            total_budget=None,
            max_history_tokens=None,
        )

    @classmethod
    def engineered(
        cls, *, total_budget: int | None = TOTAL_BUDGET, retrieval_k: int = RETRIEVAL_TOP_K
    ) -> "RunConfig":
        return cls(
            arm="engineered",
            pipeline_version="context-v4",
            routing=True,
            memory=True,
            enforce_budget=True,
            retrieval_k=retrieval_k,
            total_budget=total_budget,
            max_history_tokens=None,
        )

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RunConfig":
        if not data:
            return cls.engineered()
        return cls(**data)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    @property
    def slug(self) -> str:
        """Stable experiment label, e.g. ``engineered-b5000``."""
        if not self.enforce_budget or self.total_budget is None:
            return f"{self.arm}-b-unbounded"
        return f"{self.arm}-b{self.total_budget}"


ARMS: dict[str, RunConfig] = {
    "naive": RunConfig.naive(),
    "engineered": RunConfig.engineered(),
}

SWEEP_BUDGETS: tuple[int | None, ...] = (800, 1500, 3000, None)
