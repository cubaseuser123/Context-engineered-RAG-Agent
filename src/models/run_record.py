"""run_record - one pydantic record per agent invocation.

Records are the source of truth for every metric. Nothing in the eval path
parses a span: a field that is missing here is a field that does not exist,
instead of an empty row that reads as a wrong answer.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"


class RoutingInfo(BaseModel):
    intent: str | None = None
    reasoning: str | None = None
    department_requested: str | None = None
    department_applied: str | None = None
    needs_retrieval: bool | None = None
    search_query: str | None = None
    user_facts: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc: str
    section: str = ""
    score: float = 0.0
    vector_score: float | None = None
    tokens: int = 0


class RetrievalInfo(BaseModel):
    query: str | None = None
    k: int | None = None
    filter: str | None = None
    searches: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    # what survived the enforcer, so budget_invariant can check the drop order
    retained_chunk_ids: list[str] = Field(default_factory=list)


class MemoryInfo(BaseModel):
    read: list[str] = Field(default_factory=list)
    written: list[str] = Field(default_factory=list)
    write_error: str | None = None


class CostInfo(BaseModel):
    """Provider-reported usage; never a tiktoken estimate."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    aux_prompt_tokens: int = 0
    aux_completion_tokens: int = 0
    model: str = ""
    provider: str = "groq"


class OutputInfo(BaseModel):
    response: str = ""
    citations: list[str] = Field(default_factory=list)
    unknown_citations: list[str] = Field(default_factory=list)
    refused: bool = False


class ErrorInfo(BaseModel):
    type: str
    message: str


class RunRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION

    # identity
    run_id: str
    trace_id: str | None = None
    record_id: str
    scenario_id: str
    turn_index: int
    arm: Literal["naive", "engineered"]
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = ""
    git_sha: str | None = None
    dataset_version: str = ""

    # the turn
    question: str = ""
    started_at: str = ""
    duration_ms: float = 0.0

    routing: RoutingInfo | None = None
    retrieval: RetrievalInfo | None = None
    memory: MemoryInfo = Field(default_factory=MemoryInfo)
    budget: dict[str, Any] | None = None
    prompt: str = ""
    output: OutputInfo = Field(default_factory=OutputInfo)
    cost: CostInfo = Field(default_factory=CostInfo)
    timing: dict[str, float] = Field(default_factory=dict)
    error: ErrorInfo | None = None

    @property
    def tokens_total(self) -> int:
        return self.cost.total_tokens

    def to_jsonl(self) -> str:
        return self.model_dump_json(exclude_none=False)
