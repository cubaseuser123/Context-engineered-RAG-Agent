"""Persist explicit user facts identified by the router, without another model call."""
from __future__ import annotations

from src.models.state import AgentState
from src.run_config import RunConfig
from src.stores import memory_store


def memory_writer_node(state: AgentState) -> dict:
    config = RunConfig.from_dict(state.get("run_config"))
    if not config.memory or not state.get("user_facts"):
        return {"new_memory_entries": [], "memory_writer_usage": {}, "memory_writer_latency_ms": 0.0}
    user_id = state.get("user_id", "default")
    existing = {entry["fact"].strip().casefold() for entry in memory_store.read(user_id)}
    facts = []
    for fact in state["user_facts"]:
        if fact.strip().casefold() not in existing:
            facts.append(fact.strip())
            existing.add(fact.strip().casefold())
    memory_store.write(user_id, facts, state.get("turn_number", 0))
    memory_store.prune(user_id)
    return {"new_memory_entries": [{"fact": fact} for fact in facts],
            "memory_writer_usage": {}, "memory_writer_latency_ms": 0.0}
