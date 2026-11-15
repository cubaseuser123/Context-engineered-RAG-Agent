"""context_enforcer - budget enforcement across the four zones.

Enforcement is a parameter, not an architecture:

* ``enforce_budget=False`` (naive) passes everything through but still reports
  the token counts, so the record shows what the budget would have saved.
* ``total_budget=None`` means unbounded: the same pass-through, by definition.

Drops are typed records carrying zone, item, token cost, score and reason, which
is what makes ``budget_invariant`` checkable rather than assertable.
"""
from __future__ import annotations

import logging

from src.config import (
    CONVERSATION_BUDGET,
    MEMORY_BUDGET,
    RETRIEVAL_BUDGET,
)
from src.models.schemas import BudgetLog, BudgetZoneReport, DropRecord
from src.models.state import AgentState
from src.observability import span
from src.prompts import SYNTHESIZER_SYSTEM_PROMPT
from src.run_config import RunConfig
from src.token_utils import count_chunks, count_entries, count_messages, count_tokens

logger = logging.getLogger(__name__)

_NON_SYSTEM_BASE = RETRIEVAL_BUDGET + MEMORY_BUDGET + CONVERSATION_BUDGET


def zone_caps(config: RunConfig, system_tokens: int) -> dict[str, int | None]:
    """Zone caps, scaled together when the total budget is below the default.

    The system prompt is never trimmed, so it is subtracted from the budget
    first and the remaining three zones scale to fit what is left. At the
    default 5,000 this reproduces exactly 3,000 / 500 / 1,000.
    """
    if not config.enforce_budget or config.total_budget is None:
        return {"retrieval": None, "memory": None, "conversation": None}

    available = max(0, config.total_budget - system_tokens)
    scale = min(1.0, available / _NON_SYSTEM_BASE)
    conversation_base = (
        config.max_history_tokens
        if config.max_history_tokens is not None
        else CONVERSATION_BUDGET
    )
    return {
        "retrieval": int(RETRIEVAL_BUDGET * scale),
        "memory": int(MEMORY_BUDGET * scale),
        "conversation": int(conversation_base * scale),
    }


def _chunk_tokens(chunk: dict) -> int:
    return count_tokens(chunk.get("content", ""))


def _memory_tokens(entry: dict) -> int:
    return count_tokens(str(entry.get("fact", "")))


def _turn_tokens(turn: dict) -> int:
    return count_tokens(turn.get("role", "")) + count_tokens(turn.get("content", "")) + 4


def _total(chunks: list[dict], memory: list[dict], history: list[dict]) -> int:
    return count_chunks(chunks) + count_entries(memory, key="fact") + count_messages(history)


def context_enforcer_node(state: AgentState) -> dict:
    """Trim each zone to its cap, then trim the total, recording every drop."""
    config = RunConfig.from_dict(state.get("run_config"))
    chunks = list(state.get("retrieved_chunks", []))
    # memory_store.read() returns newest first, so pop() drops the oldest fact.
    memory = list(state.get("memory_entries", []))
    history = list(state.get("conversation_history", []))

    system_tokens = count_tokens(SYNTHESIZER_SYSTEM_PROMPT)
    caps = zone_caps(config, system_tokens)
    total_cap = (
        system_tokens + sum(v for v in caps.values() if v is not None)
        if any(v is not None for v in caps.values())
        else None
    )

    before = {
        "retrieval": count_chunks(chunks),
        "memory": count_entries(memory, key="fact"),
        "conversation": count_messages(history),
    }
    total_before = system_tokens + sum(before.values())
    drops: list[DropRecord] = []

    retrieval_cap = caps["retrieval"]
    if retrieval_cap is not None and before["retrieval"] > retrieval_cap:
        chunks.sort(key=lambda c: c.get("score", 0.0))
        while chunks and count_chunks(chunks) > retrieval_cap:
            dropped = chunks.pop(0)
            drops.append(
                DropRecord(
                    zone="retrieval",
                    item_id=dropped.get("chunk_id", ""),
                    tokens=_chunk_tokens(dropped),
                    reason="lowest_score",
                    doc=dropped.get("source_doc"),
                    score=dropped.get("score"),
                )
            )

    conversation_cap = caps["conversation"]
    if conversation_cap is not None and count_messages(history) > conversation_cap:
        while history and count_messages(history) > conversation_cap:
            dropped = history.pop(0)  # oldest turn first
            drops.append(
                DropRecord(
                    zone="conversation",
                    item_id=f"{dropped.get('role', '?')}:{dropped.get('content', '')[:40]}",
                    tokens=_turn_tokens(dropped),
                    reason="oldest_turn",
                )
            )

    memory_cap = caps["memory"]
    if memory_cap is not None and count_entries(memory, key="fact") > memory_cap:
        while memory and count_entries(memory, key="fact") > memory_cap:
            dropped = memory.pop()  # oldest fact (list is newest first)
            drops.append(
                DropRecord(
                    zone="memory",
                    item_id=str(dropped.get("fact", ""))[:60],
                    tokens=_memory_tokens(dropped),
                    reason="oldest_fact",
                )
            )

    if total_cap is not None:
        while system_tokens + _total(chunks, memory, history) > total_cap:
            if chunks:
                chunks.sort(key=lambda c: c.get("score", 0.0))
                dropped = chunks.pop(0)
                drops.append(
                    DropRecord(
                        zone="retrieval",
                        item_id=dropped.get("chunk_id", ""),
                        tokens=_chunk_tokens(dropped),
                        reason="over_total",
                        doc=dropped.get("source_doc"),
                        score=dropped.get("score"),
                    )
                )
            elif history:
                dropped = history.pop(0)
                drops.append(
                    DropRecord(
                        zone="conversation",
                        item_id=f"{dropped.get('role', '?')}:{dropped.get('content', '')[:40]}",
                        tokens=_turn_tokens(dropped),
                        reason="over_total",
                    )
                )
            elif memory:
                dropped = memory.pop()
                drops.append(
                    DropRecord(
                        zone="memory",
                        item_id=str(dropped.get("fact", ""))[:60],
                        tokens=_memory_tokens(dropped),
                        reason="over_total",
                    )
                )
            else:
                break

    after = {
        "retrieval": count_chunks(chunks),
        "memory": count_entries(memory, key="fact"),
        "conversation": count_messages(history),
    }
    total_after = system_tokens + sum(after.values())

    zones = [
        BudgetZoneReport(
            zone_name="system_prompt",
            cap=system_tokens,
            tokens_before=system_tokens,
            tokens_after=system_tokens,
            items_dropped=0,
            dropped=[],
        )
    ]
    for zone_name in ("retrieval", "memory", "conversation"):
        zone_drops = [d for d in drops if d.zone == zone_name]
        zones.append(
            BudgetZoneReport(
                zone_name=zone_name,
                cap=caps[zone_name],
                tokens_before=before[zone_name],
                tokens_after=after[zone_name],
                items_dropped=len(zone_drops),
                dropped=zone_drops,
            )
        )

    budget_log = BudgetLog(
        zones=zones,
        total_before=total_before,
        total_after=total_after,
        total_cap=total_cap,
        enforced=bool(drops),
    )

    with span(
        "node.context_enforcer",
        **{
            "node.name": "context_enforcer",
            "arm": config.arm,
            "budget.total_budget": config.total_budget,
            "budget.total_before": total_before,
            "budget.total_after": total_after,
            "budget.enforced": budget_log.enforced,
            "budget.drops": [d.model_dump() for d in drops],
        },
    ):
        pass

    logger.info(
        "Context enforcer: %d->%d tokens, %d drops (arm=%s, budget=%s)",
        total_before,
        total_after,
        len(drops),
        config.arm,
        config.total_budget,
    )

    return {
        "trimmed_chunks": chunks,
        "trimmed_memory": memory,
        "trimmed_history": history,
        "budget_log": budget_log.model_dump(),
    }
