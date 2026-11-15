from src.nodes.context_enforcer import context_enforcer_node
from src.run_config import RunConfig


def _chunk(index: int, score: float, words: int = 900) -> dict:
    return {
        "chunk_id": f"HR_Leave_Policy::{index:03d}",
        "source_doc": "HR_Leave_Policy",
        "section": "Carry-forward",
        "department": "HR",
        "score": score,
        "content": "alpha " * words,
    }


def _state(chunks, memory=None, history=None, config=None) -> dict:
    return {
        "retrieved_chunks": chunks,
        "memory_entries": memory or [],
        "conversation_history": history or [],
        "run_config": (config or RunConfig.engineered()).as_dict(),
    }


def test_drops_the_lowest_scoring_chunks_first():
    chunks = [_chunk(i, score=0.1 * (i + 1)) for i in range(8)]
    result = context_enforcer_node(_state(chunks))

    trimmed = result["trimmed_chunks"]
    dropped = [
        item
        for zone in result["budget_log"]["zones"]
        if zone["zone_name"] == "retrieval"
        for item in zone["dropped"]
    ]
    assert len(trimmed) < 8
    assert dropped, "8 x ~900 words must overrun the 3,000-token retrieval cap"
    assert max(d["score"] for d in dropped) <= min(c["score"] for c in trimmed)
    assert all(d["reason"] == "lowest_score" for d in dropped)
    assert result["budget_log"]["enforced"] is True


def test_every_zone_stays_within_its_cap():
    chunks = [_chunk(i, score=0.1 * (i + 1)) for i in range(8)]
    result = context_enforcer_node(
        _state(
            chunks,
            memory=[{"fact": "word " * 400, "timestamp": "t", "source_turn": 1}],
            history=[{"role": "user", "content": "word " * 400} for _ in range(6)],
        )
    )
    for zone in result["budget_log"]["zones"]:
        if zone["cap"] is not None:
            assert zone["tokens_after"] <= zone["cap"], zone["zone_name"]
    total_cap = result["budget_log"]["total_cap"]
    assert total_cap is not None
    assert result["budget_log"]["total_after"] <= total_cap


def test_memory_is_newest_first_so_the_oldest_fact_is_dropped():
    # memory_store.read() returns newest first; the enforcer pops the tail.
    memory = [
        {"fact": "NEWEST " + "word " * 300, "timestamp": "4", "source_turn": 4},
        {"fact": "SECOND " + "word " * 300, "timestamp": "3", "source_turn": 3},
        {"fact": "THIRD " + "word " * 300, "timestamp": "2", "source_turn": 2},
        {"fact": "OLDEST " + "word " * 300, "timestamp": "1", "source_turn": 1},
    ]
    result = context_enforcer_node(_state([], memory=memory))
    kept = [m["fact"].split()[0] for m in result["trimmed_memory"]]
    assert "OLDEST" not in kept
    assert kept[0] == "NEWEST"


def test_conversation_drops_oldest_turns():
    history = [{"role": "user", "content": f"turn {i} " + "word " * 300} for i in range(6)]
    result = context_enforcer_node(_state([], history=history))
    assert result["trimmed_history"][0]["content"].startswith("turn 3")
    assert result["trimmed_history"][-1]["content"].startswith("turn 5")


def test_naive_arm_passes_everything_and_reports_the_saving():
    chunks = [_chunk(i, score=0.1 * (i + 1)) for i in range(8)]
    history = [{"role": "user", "content": "word " * 900} for _ in range(4)]
    result = context_enforcer_node(
        _state(chunks, history=history, config=RunConfig.naive())
    )
    assert len(result["trimmed_chunks"]) == 8
    assert len(result["trimmed_history"]) == 4
    assert result["budget_log"]["enforced"] is False
    assert result["budget_log"]["total_cap"] is None
    assert result["budget_log"]["total_after"] == result["budget_log"]["total_before"]


def test_unbounded_engineered_budget_does_not_trim():
    chunks = [_chunk(i, score=0.1 * (i + 1)) for i in range(8)]
    config = RunConfig.engineered(total_budget=None)
    result = context_enforcer_node(_state(chunks, config=config))
    assert len(result["trimmed_chunks"]) == 8
    assert result["budget_log"]["total_cap"] is None
