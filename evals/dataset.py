"""dataset - versioned scenarios with ground truth, validated before a run.

Document IDs are canonical and extension-free; validation fails loudly when an
expected document does not exist in the corpus, because a typo in ground truth
is indistinguishable from a retrieval miss at scoring time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

COHORTS = ("single_doc", "trap", "multi_turn", "memory", "refusal")
SINGLE_TURN_COHORTS = ("single_doc", "trap")


class Expected(BaseModel):
    intent: str | None = None
    docs: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    answer_type: Literal["numeric", "date", "rule", "refusal", "free"] = "free"
    refusal_warranted: bool = False
    reference_answer: str = ""


class Turn(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str
    expected: Expected | None = None


class Setup(BaseModel):
    user_id: str = "eval_user"
    history: list[Turn] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    id: str
    cohort: Literal["single_doc", "trap", "multi_turn", "memory", "refusal"]
    tags: list[str] = Field(default_factory=list)
    turns: list[Turn]
    setup: Setup = Field(default_factory=Setup)
    expected: Expected = Field(default_factory=Expected)

    @field_validator("turns")
    @classmethod
    def _needs_a_user_turn(cls, turns: list[Turn]) -> list[Turn]:
        if not any(t.role == "user" for t in turns):
            raise ValueError("scenario has no user turn")
        return turns

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def expected_turns(self) -> int:
        return len(self.user_turns)

    def expected_for(self, user_turn_index: int) -> Expected | None:
        """Per-turn ground truth, falling back to the scenario level.

        The scenario-level ``expected`` applies to the final user turn only;
        earlier turns must carry their own expectation or they are scored on
        health and tokens alone.
        """
        user_turns = self.user_turns
        if user_turn_index < 0 or user_turn_index >= len(user_turns):
            return None
        turn = user_turns[user_turn_index]
        if turn.expected is not None:
            return turn.expected
        if user_turn_index == len(user_turns) - 1:
            return self.expected
        return None


class Dataset(BaseModel):
    version: str
    created: str = ""
    scenarios: list[Scenario]

    @property
    def hash(self) -> str:
        blob = json.dumps(
            [s.model_dump() for s in self.scenarios], sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def by_id(self, scenario_id: str) -> Scenario:
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        raise KeyError(scenario_id)

    @property
    def turns_per_arm(self) -> int:
        return sum(s.expected_turns for s in self.scenarios)

    def validate_against(self, corpus_ids: set[str]) -> list[str]:
        """Return a list of problems; empty means the dataset is usable."""
        problems: list[str] = []
        seen: set[str] = set()
        for scenario in self.scenarios:
            if scenario.id in seen:
                problems.append(f"duplicate scenario id: {scenario.id}")
            seen.add(scenario.id)

            expectations = [scenario.expected] + [
                t.expected for t in scenario.turns if t.expected is not None
            ]
            for expected in expectations:
                for doc in expected.docs:
                    if doc not in corpus_ids:
                        problems.append(
                            f"{scenario.id}: expected doc '{doc}' is not in the corpus"
                        )

            if scenario.cohort == "multi_turn" and scenario.expected_turns < 4:
                problems.append(
                    f"{scenario.id}: multi_turn scenario has only "
                    f"{scenario.expected_turns} user turns"
                )
            if scenario.cohort == "refusal":
                if not scenario.expected.refusal_warranted:
                    problems.append(
                        f"{scenario.id}: refusal cohort without refusal_warranted"
                    )
            if scenario.cohort == "memory" and not scenario.setup.memory and not (
                scenario.expected_turns > 1
            ):
                problems.append(
                    f"{scenario.id}: memory scenario needs seeded memory or multiple turns"
                )
        return problems

    def cohort_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for scenario in self.scenarios:
            counts[scenario.cohort] = counts.get(scenario.cohort, 0) + 1
        return counts


def load_dataset(path: Path) -> Dataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Dataset(**data)
