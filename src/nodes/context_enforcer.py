"""
context_enforcer - Budget enforcement node.
Takes: retrieved_chunks, memory_entries, conversation_history.
Returns: trimmed_chunks, trimmed_memory, trimmed_history, budget_log.
"""
import logging 

from src.config import(
    CONVERSATION_BUDGET,
    MEMORY_BUDGET,
    RETRIEVAL_BUDGET,
    SYSTEM_PROMPT_BUDGET,
    TOTAL_BUDGET,
)
from src.models.state import AgentState
from src.token_utils import count_chunks, count_entries, count_messages

logger = logging.getLogger(__name__)

def get_chunk_score(chunk):
    return chunk.get("score", 0.0)

def context_enforcer_node(state: AgentState) -> dict:
    """
    Enforce the context budget across all 4 zones.
    """
    chunks = list(state.get("retrieved_chunks", []))
    memory = list(state.get("memory_entries", []))
    history = list(state.get("conversation_history", []))

    drop_details = []
    enforced = False 

    system_tokens = SYSTEM_PROMPT_BUDGET
    retrieval_tokens = count_chunks(chunks)
    memory_tokens = count_entries(memory, key="fact")
    conversation_tokens = count_messages(history)

    total_before = system_tokens + retrieval_tokens + memory_tokens + conversation_tokens

    if retrieval_tokens > RETRIEVAL_BUDGET:
        enforced = True 
    
        chunks.sort(key=get_chunk_score)

        while chunks and count_chunks(chunks) > RETRIEVAL_BUDGET:
            dropped = chunks.pop(0)
            drop_details.append(f"Dropped retrieval chunk from '{dropped.get('source_doc', '?')}'")
        retrieval_tokens = count_chunks(chunks)

    if conversation_tokens > CONVERSATION_BUDGET:
        enforced = True
        while history and count_messages(history) > CONVERSATION_BUDGET:
            dropped = history.pop(0)
            drop_details.append(f"Dropped conversation turn: {dropped.get('role', '?')}")
        conversation_tokens = count_messages(history)

    if memory_tokens > MEMORY_BUDGET:
        enforced = True
        while memory and count_entries(memory, key="fact") > MEMORY_BUDGET:
            dropped = memory.pop()
            drop_details.append(f"Dropped memory entry")
        memory_tokens = count_entries(memory, key="fact")

    total_after = system_tokens + retrieval_tokens + memory_tokens + conversation_tokens

    #now if we are still over budget after this 
    if total_after > TOTAL_BUDGET:
        enforced = True
        chunks.sort(key=get_chunk_score)

        while chunks and (system_tokens + count_chunks(chunks) + memory_tokens + conversation_tokens) > TOTAL_BUDGET:
            dropped = chunks.pop(0)
            drop_details.append(f"Emergency drop: retrieval chunks from '{dropped.get('source_doc', '?')}'")

        retrieval_tokens = count_chunks(chunks)
        total_after = system_tokens + retrieval_tokens + memory_tokens + conversation_tokens


    #clear logs. this will help us with observability later 
    retrieval_dropped = 0 
    memory_dropped = 0 
    conversation_dropped = 0 

    for detail in drop_details:
        if "retrieval" in detail.lower():
            retrieval_dropped += 1
        if "memory" in detail.lower():
            memory_dropped += 1
        if "conversation" in detail.lower():
            conversation_dropped += 1

    budget_log = {
        "zones" : [
            {"zone_name" : "system_prompt", "token_count": system_tokens, "budget": SYSTEM_PROMPT_BUDGET,"items_dropped": 0},
            {"zone_name": "retrieval", "token_count": retrieval_tokens, "budget": RETRIEVAL_BUDGET, "items_dropped": retrieval_dropped},
            {"zone_name": "memory", "token_count": memory_tokens, "budget": MEMORY_BUDGET, "items_dropped": memory_dropped},
            {"zone_name": "conversation", "token_count": conversation_tokens, "budget": CONVERSATION_BUDGET, "items_dropped": conversation_dropped},
        ],
        "total_before" : total_before,
        "total_after" : total_after,
        "enforced" : enforced,
        "drop_details" : drop_details,
    }
    
    if enforced:
        logger.info("Context enforcer: trimmed %d->%d tokens, %d items dropped", total_before, total_after, len(drop_details))
    else:
        logger.info("Context enforcer: no trimming needed (%d tokens)", total_after)

    return{
        "trimmed_chunks" : chunks,
        "trimmed_memory" : memory,
        "trimmed_history" : history,
        "budget_log" : budget_log,
    }