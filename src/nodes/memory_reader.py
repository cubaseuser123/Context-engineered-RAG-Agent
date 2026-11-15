"""memory_reader - fetch the user's stored facts.

No LLM call. The naive arm has no memory, so it records an empty zone rather
than reading rows it will not use.
"""
from __future__ import annotations

import logging

from src.models.state import AgentState
from src.run_config import RunConfig
from src.stores import memory_store

logger = logging.getLogger(__name__)


def memory_reader_node(state: AgentState) -> dict:
    config = RunConfig.from_dict(state.get("run_config"))
    if not config.memory:
        return {"memory_entries": []}

    user_id = state.get("user_id", "default")
    entries = memory_store.read(user_id)
    logger.info("Memory reader: %d entries for user '%s'", len(entries), user_id)
    return {"memory_entries": entries}
