# Context Engineering Code Reference

> [!IMPORTANT]
> These files depend on modules from the **Agent Code Ref** (config, state, schemas, vector_store). Build the Agent Code Ref first.

**Build order**: token_utils → context_enforcer → ingest_documents

---

### `src/token_utils.py`

Tiktoken-based token counting and truncation utilities. Used by the context enforcer to measure and trim each zone.

```python
"""
token_utils — Token counting and truncation using tiktoken.
Provides budget-aware utilities for the context enforcer node.
Uses cl100k_base encoding (~5% variance from Gemini's actual tokenizer — acceptable for Phase 1).
"""
import logging

import tiktoken

logger = logging.getLogger(__name__)

# Singleton encoder — loaded once, reused everywhere
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def count_messages(messages: list[dict]) -> int:
    """Count total tokens across a list of message dicts [{role, content}]."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("role", ""))
        total += count_tokens(msg.get("content", ""))
        total += 4  # overhead per message (role markers, separators)
    return total


def count_entries(entries: list[dict], key: str = "fact") -> int:
    """Count total tokens across a list of dicts by a specific text key."""
    total = 0
    for entry in entries:
        total += count_tokens(str(entry.get(key, "")))
    return total


def count_chunks(chunks: list[dict]) -> int:
    """Count total tokens across retrieved chunks (uses 'content' field)."""
    return sum(count_tokens(c.get("content", "")) for c in chunks)


def truncate_to_budget(text: str, budget: int) -> str:
    """Truncate text to fit within a token budget. Returns the truncated string."""
    if not text:
        return text
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= budget:
        return text
    truncated = enc.decode(tokens[:budget])
    logger.debug("Truncated text from %d to %d tokens", len(tokens), budget)
    return truncated
```

> [!NOTE]
> `cl100k_base` is the tiktoken encoding for GPT-4/ChatGPT. It's ~5% off from Gemini's tokenizer. For budget enforcement this is close enough. To use Gemini's exact tokenizer, swap to `google.generativeai.count_tokens()` — but that's an API call per count, which adds latency.

---

### `src/nodes/context_enforcer.py`

The core context engineering node. Pure deterministic logic — no LLM call. Counts tokens across all 4 zones, trims if over budget, and logs every drop action.

```python
"""
context_enforcer — Budget enforcement node.
Takes: retrieved_chunks, memory_entries, conversation_history.
Returns: trimmed_chunks, trimmed_memory, trimmed_history, budget_log.
No LLM call — pure deterministic token counting and trimming.
"""
import logging

from src.config import (
    CONVERSATION_BUDGET,
    MEMORY_BUDGET,
    RETRIEVAL_BUDGET,
    SYSTEM_PROMPT_BUDGET,
    TOTAL_BUDGET,
)
from src.models.state import AgentState
from src.token_utils import count_chunks, count_entries, count_messages

logger = logging.getLogger(__name__)


def context_enforcer_node(state: AgentState) -> dict:
    """
    Enforce the context budget across all 4 zones.

    Trimming priority (least valuable first):
    1. Drop lowest-scoring retrieval chunks
    2. Truncate oldest conversation history turns
    3. Drop oldest memory entries

    The system prompt zone is fixed and never trimmed.
    """
    # Get raw inputs
    chunks = list(state.get("retrieved_chunks", []))
    memory = list(state.get("memory_entries", []))
    history = list(state.get("conversation_history", []))

    # --- Track state for the budget log ---
    drop_details: list[str] = []
    enforced = False

    # --- Measure initial token counts per zone ---
    system_tokens = SYSTEM_PROMPT_BUDGET  # fixed, we don't trim this
    retrieval_tokens = count_chunks(chunks)
    memory_tokens = count_entries(memory, key="fact")
    conversation_tokens = count_messages(history)

    total_before = system_tokens + retrieval_tokens + memory_tokens + conversation_tokens

    # --- Step A: Trim retrieval zone ---
    if retrieval_tokens > RETRIEVAL_BUDGET:
        enforced = True
        # Sort by score ascending (worst first), drop until under budget
        chunks.sort(key=lambda c: c.get("score", 0.0))
        while chunks and count_chunks(chunks) > RETRIEVAL_BUDGET:
            dropped = chunks.pop(0)
            drop_details.append(
                f"Dropped retrieval chunk from '{dropped.get('source_doc', '?')}' "
                f"(score={dropped.get('score', 0):.3f})"
            )
        retrieval_tokens = count_chunks(chunks)

    # --- Step B: Trim conversation history ---
    if conversation_tokens > CONVERSATION_BUDGET:
        enforced = True
        # Drop oldest turns first (keep most recent)
        while history and count_messages(history) > CONVERSATION_BUDGET:
            dropped = history.pop(0)
            drop_details.append(
                f"Dropped conversation turn: {dropped.get('role', '?')}: "
                f"'{dropped.get('content', '')[:50]}...'"
            )
        conversation_tokens = count_messages(history)

    # --- Step C: Trim memory ---
    if memory_tokens > MEMORY_BUDGET:
        enforced = True
        # Drop oldest entries first (they're already sorted by timestamp DESC from memory_reader)
        while memory and count_entries(memory, key="fact") > MEMORY_BUDGET:
            dropped = memory.pop()  # pop from end = oldest
            drop_details.append(f"Dropped memory entry: '{dropped.get('fact', '')[:50]}...'")
        memory_tokens = count_entries(memory, key="fact")

    # --- Final total check ---
    total_after = system_tokens + retrieval_tokens + memory_tokens + conversation_tokens

    if total_after > TOTAL_BUDGET:
        enforced = True
        # Emergency trimming: drop more retrieval chunks
        chunks.sort(key=lambda c: c.get("score", 0.0))
        while chunks and (system_tokens + count_chunks(chunks) + memory_tokens + conversation_tokens) > TOTAL_BUDGET:
            dropped = chunks.pop(0)
            drop_details.append(
                f"Emergency drop: retrieval chunk from '{dropped.get('source_doc', '?')}'"
            )
        retrieval_tokens = count_chunks(chunks)
        total_after = system_tokens + retrieval_tokens + memory_tokens + conversation_tokens

    # --- Build budget log ---
    budget_log = {
        "zones": [
            {"zone_name": "system_prompt", "token_count": system_tokens, "budget": SYSTEM_PROMPT_BUDGET, "items_dropped": 0},
            {
                "zone_name": "retrieval",
                "token_count": retrieval_tokens,
                "budget": RETRIEVAL_BUDGET,
                "items_dropped": len([d for d in drop_details if "retrieval" in d.lower()]),
            },
            {
                "zone_name": "memory",
                "token_count": memory_tokens,
                "budget": MEMORY_BUDGET,
                "items_dropped": len([d for d in drop_details if "memory" in d.lower()]),
            },
            {
                "zone_name": "conversation",
                "token_count": conversation_tokens,
                "budget": CONVERSATION_BUDGET,
                "items_dropped": len([d for d in drop_details if "conversation" in d.lower()]),
            },
        ],
        "total_before": total_before,
        "total_after": total_after,
        "enforced": enforced,
        "drop_details": drop_details,
    }

    if enforced:
        logger.info(
            "Context enforcer: trimmed %d→%d tokens, %d items dropped",
            total_before, total_after, len(drop_details),
        )
    else:
        logger.info("Context enforcer: no trimming needed (%d tokens)", total_after)

    return {
        "trimmed_chunks": chunks,
        "trimmed_memory": memory,
        "trimmed_history": history,
        "budget_log": budget_log,
    }
```

> [!TIP]
> The `budget_log` dict is the key artifact for Phoenix observability. When traces are viewed in the Phoenix UI, this log shows exactly what was dropped and why — making context engineering decisions visible, not just claimed.

---

### `scripts/__init__.py`

```python
"""scripts — CLI utilities for corpus management and testing."""
```

### `scripts/ingest_documents.py`

CLI script to chunk and embed all policy documents into Pinecone. Run once after corpus generation.

```python
"""
ingest_documents — CLI script to ingest policy documents into Pinecone.
Reads all .txt files from corpus/documents/, chunks, embeds, and upserts.
Run: python -m scripts.ingest_documents
"""
import logging
import sys

from src.config import CORPUS_DIR
from src.stores.vector_store import ingest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    if not CORPUS_DIR.exists():
        logger.error("Corpus directory not found: %s", CORPUS_DIR)
        logger.error("Generate the corpus first, then run this script.")
        sys.exit(1)

    doc_count = len(list(CORPUS_DIR.glob("*.txt")))
    if doc_count == 0:
        logger.error("No .txt files found in %s", CORPUS_DIR)
        sys.exit(1)

    logger.info("Starting ingestion of %d documents from %s", doc_count, CORPUS_DIR)
    total_chunks = ingest(CORPUS_DIR)
    logger.info("Done. %d total chunks upserted to Pinecone.", total_chunks)


if __name__ == "__main__":
    main()
```

---

## Changes from Original Spec

| Area | Spec | Changed To | Why |
|---|---|---|---|
| Token counting | Gemini `count_tokens()` API | tiktoken `cl100k_base` | Local, fast, no API call per count. ~5% variance acceptable |
| Emergency trimming | Not specified | Added final-pass retrieval drop | Safety valve if per-zone trimming isn't enough |
| Budget log format | Generic dict | Structured with zone reports | Makes Phoenix traces more readable |
| Ingest script | Part of vector_store | Separate CLI script | Clean separation of one-time ingestion from runtime query |
