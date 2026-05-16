from __future__ import annotations
from typing import Any 
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    user_id: str
    query: str
    conversation_history: list[dict[str, str]]
    turn_number: int 

    #router output
    intent: str
    router_reasoning: str
    retrieval_metadata_filter: str | None

    #retrieval output
    retrieved_chunks: list[dict[str, Any]]

    #memory reader 
    memory_entries: list[dict[str, Any]]

    #context enforcer output
    trimmed_chunks: list[dict[str, Any]]
    trimmed_history: list[dict[str, str]]
    trimmed_memory: list[dict[str, Any]]
    budget_log: dict[str, Any]

    #synthesis output
    response: str
    cited_sources: list[str]

    #memory writer output
    new_memory_entries: list[dict[str, Any]]
