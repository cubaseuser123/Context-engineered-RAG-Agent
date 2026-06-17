"""
memory_reader - Fetch user facts from SQLite memory store.
Takes : user_id. Returns: memory_entries.
Runs for all the intents except out_of_scope. No LLM call.
""" 
import logging 

from src.models.state import AgentState
from src.stores import memory_store

logger = logging.getLogger(__name__)

def memory_reader_node(state: AgentState) -> dict:
    """Read all memory entries for the current user."""
    user_id = state.get("user_id", "default")
    entries = memory_store.read(user_id)
    logger.info("Memory reader: %d entries for user '%s'", len(entries), user_id)
    return {"memory_entries" : entries}