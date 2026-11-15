from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    needs_retrieval: bool = True
    search_query: str = ""
    user_facts: list[str] = Field(default_factory=list)
    intent: str = Field(
        description="One of: policy_lookup, clarification, memory_recall, out_of_scope"
    )
    reasoning: str = Field(description="Why this intent was selected")
    department_filter: str | None = Field(
        default=None, description="Department for metadata filtering"
    )


class MemoryEntry(BaseModel):
    fact: str
    timestamp: str
    source_turn: int


class RetrievalChunk(BaseModel):
    chunk_id: str
    content: str
    source_doc: str
    section: str
    department: str
    score: float


class DropRecord(BaseModel):
    """One item the enforcer removed, with why and how much it cost."""

    zone: Literal["retrieval", "memory", "conversation"]
    item_id: str
    tokens: int
    reason: Literal["lowest_score", "oldest_turn", "oldest_fact", "over_total"]
    doc: str | None = None
    score: float | None = None


class BudgetZoneReport(BaseModel):
    zone_name: str
    cap: int | None
    tokens_before: int
    tokens_after: int
    items_dropped: int = 0
    dropped: list[DropRecord] = Field(default_factory=list)


class BudgetLog(BaseModel):
    zones: list[BudgetZoneReport] = Field(default_factory=list)
    total_before: int = 0
    total_after: int = 0
    total_cap: int | None = None
    enforced: bool = False
