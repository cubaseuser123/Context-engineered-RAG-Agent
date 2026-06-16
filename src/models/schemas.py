from __future__ import annotations
from pydantic import BaseModel, Field

class RouterOutput(BaseModel):
    intent: str = Field(description= 'One of: policy_lookup, clarification, memory_recall, out_of_scope')
    reasoning: str = Field(description="Why this intent was selected")
    department_filter: str | None = Field(default=None, description="Department for metadata filtering")

class MemoryEntry(BaseModel):
    fact: str 
    timestamp: str
    source_turn:  int

class RetrievalChunk(BaseModel):
    content: str
    source_doc: str
    section: str 
    department: str 
    score:float 

class BudgetZoneReport(BaseModel):
    zone_name: str 
    token_count: int 
    budget: int
    items_dropped: int = 0

class BudgetLog(BaseModel):
    zones: list[BudgetZoneReport] = Field(default_factory = list)
    total_before: int = 0
    total_after: int = 0
    enforced: bool = False
    drop_details: list[str] = Field(default_factory=list)

class TestQuestion(BaseModel):
    id: int
    question: str
    category: str
    expected_intent: str
    expected_sources: list[str]
    expected_answer: str 
    triggers_budget_enforcement: bool
    correct_answer_is_idk: bool

class ScorecardEntry(BaseModel):
    question_id: int
    router_correct: bool
    source_correct: bool
    answer_correct: bool | None = None
    hallucinated: bool = False

