# Observability Code Reference

> [!IMPORTANT]
> These files depend on modules from both the **Agent Code Ref** and **Context Engineering Code Ref**. Build those first.

**Build order**: tracing → run_test_suite → tests

---

### `src/tracing.py`

Phoenix + OpenTelemetry bootstrap. Called once at app startup. Auto-instruments every LangGraph node via `LangChainInstrumentor`.

```python
"""
tracing — Arize Phoenix + OpenTelemetry instrumentation bootstrap.
Call init_tracing() once at app startup before any LangGraph invocations.
Every node execution becomes a span in Phoenix automatically.
"""
import logging

import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from src.config import PHOENIX_ENDPOINT, PHOENIX_PROJECT_NAME

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing() -> None:
    """
    Start Phoenix locally and configure OpenTelemetry to send traces to it.
    Safe to call multiple times — only initializes once.
    """
    global _initialized
    if _initialized:
        return

    # Launch Phoenix UI on localhost:6006
    px.launch_app()
    logger.info("Phoenix launched at http://localhost:6006")

    # Configure OTEL tracer provider
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT))
    )
    trace_api.set_tracer_provider(tracer_provider)

    # Auto-instrument LangChain (covers LangGraph nodes automatically)
    LangChainInstrumentor().instrument()

    _initialized = True
    logger.info("Tracing initialized: OTEL → Phoenix at %s", PHOENIX_ENDPOINT)


def get_tracer(name: str = PHOENIX_PROJECT_NAME):
    """Get a named tracer for custom spans (e.g. budget enforcement logging)."""
    return trace_api.get_tracer(name)
```

> [!NOTE]
> `LangChainInstrumentor` auto-instruments LangGraph — no separate LangGraph instrumentor needed. Every node function call becomes a span with inputs/outputs captured automatically.

> [!TIP]
> After running the agent, open `http://localhost:6006` in your browser. Click any trace to see the full node-by-node execution tree with token counts, latencies, and the `budget_log` from the context enforcer.

---

### `scripts/run_test_suite.py`

Runs all 30 test questions against the compiled agent and produces a scorecard. Scores router accuracy, source accuracy, and hallucination rate automatically. Answer correctness is left for manual scoring on first pass.

```python
"""
run_test_suite — Execute the 30-question test suite and produce a scorecard.
Scores: router accuracy, source accuracy, hallucination rate.
Answer correctness is marked None for manual human review on first pass.
Run: python -m scripts.run_test_suite
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import TEST_SUITE_PATH, PROJECT_ROOT
from src.graph import compile_agent, run_query
from src.tracing import init_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_suite() -> list[dict]:
    """Load test questions from corpus/test_suite.json."""
    if not TEST_SUITE_PATH.exists():
        logger.error("Test suite not found: %s", TEST_SUITE_PATH)
        sys.exit(1)
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def score_question(question: dict, result: dict) -> dict:
    """Score a single question against the agent's result."""
    # Router accuracy
    actual_intent = result.get("intent", "")
    router_correct = actual_intent == question["expected_intent"]

    # Source accuracy
    cited = set(result.get("cited_sources", []))
    expected = set(question["expected_sources"])
    source_correct = bool(expected & cited) if expected else True

    # Hallucination check (for IDK questions)
    hallucinated = False
    if question.get("correct_answer_is_idk", False):
        response = result.get("response", "").lower()
        idk_phrases = ["i don't have", "not in the available", "i can't find", "no information"]
        has_idk = any(phrase in response for phrase in idk_phrases)
        hallucinated = not has_idk

    return {
        "question_id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected_intent": question["expected_intent"],
        "actual_intent": actual_intent,
        "router_correct": router_correct,
        "expected_sources": list(expected),
        "actual_sources": list(cited),
        "source_correct": source_correct,
        "answer_correct": None,  # manual scoring on first pass
        "hallucinated": hallucinated,
        "response_preview": result.get("response", "")[:200],
        "budget_enforced": result.get("budget_log", {}).get("enforced", False),
    }


def compute_aggregate(results: list[dict]) -> dict:
    """Compute aggregate scorecard from individual question scores."""
    total = len(results)
    if total == 0:
        return {}

    router_correct = sum(1 for r in results if r["router_correct"])
    source_correct = sum(1 for r in results if r["source_correct"])
    hallucinated = sum(1 for r in results if r["hallucinated"])
    budget_enforced = sum(1 for r in results if r["budget_enforced"])

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "router_correct": 0, "source_correct": 0}
        categories[cat]["total"] += 1
        if r["router_correct"]:
            categories[cat]["router_correct"] += 1
        if r["source_correct"]:
            categories[cat]["source_correct"] += 1

    return {
        "total_questions": total,
        "router_accuracy": round(router_correct / total * 100, 1),
        "source_accuracy": round(source_correct / total * 100, 1),
        "hallucination_rate": round(hallucinated / total * 100, 1),
        "hallucination_count": hallucinated,
        "budget_enforcement_count": budget_enforced,
        "per_category": categories,
    }


def save_results(results: list[dict], aggregate: dict) -> Path:
    """Save scorecard to a JSON file."""
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"scorecard_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "aggregate": aggregate,
        "questions": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path


def print_scorecard(aggregate: dict) -> None:
    """Print a human-readable scorecard to stdout."""
    print("\n" + "=" * 60)
    print("  SCORECARD")
    print("=" * 60)
    print(f"  Total questions:        {aggregate['total_questions']}")
    print(f"  Router accuracy:        {aggregate['router_accuracy']}%")
    print(f"  Source accuracy:         {aggregate['source_accuracy']}%")
    print(f"  Hallucination rate:     {aggregate['hallucination_rate']}%")
    print(f"  Budget enforcements:    {aggregate['budget_enforcement_count']}")
    print("-" * 60)
    print("  Per-category breakdown:")
    for cat, data in aggregate.get("per_category", {}).items():
        router_pct = round(data["router_correct"] / data["total"] * 100, 1)
        source_pct = round(data["source_correct"] / data["total"] * 100, 1)
        print(f"    {cat}: router={router_pct}%, source={source_pct}% (n={data['total']})")
    print("=" * 60 + "\n")


def main():
    # Initialize tracing so all runs appear in Phoenix
    init_tracing()

    # Load test suite
    questions = load_test_suite()
    logger.info("Loaded %d test questions", len(questions))

    # Compile agent
    app = compile_agent()

    # Run all questions
    results = []
    for i, q in enumerate(questions):
        logger.info("[%d/%d] Running: %s", i + 1, len(questions), q["question"][:60])
        try:
            result = run_query(app, query=q["question"], user_id="test_user", turn=i)
            scored = score_question(q, result)
            results.append(scored)
        except Exception as e:
            logger.error("[%d/%d] Failed: %s", i + 1, len(questions), str(e))
            results.append({
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "router_correct": False,
                "source_correct": False,
                "hallucinated": False,
                "answer_correct": None,
                "error": str(e),
            })

    # Compute and display scorecard
    aggregate = compute_aggregate(results)
    print_scorecard(aggregate)

    # Save to file
    output_path = save_results(results, aggregate)
    logger.info("Scorecard saved to %s", output_path)
    logger.info("View traces at http://localhost:6006")


if __name__ == "__main__":
    main()
```

---

### `tests/test_context_enforcer.py`

Unit tests for the context budget enforcer. Tests under-budget passthrough, retrieval trimming, conversation trimming, memory trimming, and emergency trimming.

```python
"""
test_context_enforcer — Unit tests for the context budget enforcer node.
Tests trimming logic without any LLM calls.
Run: python -m pytest tests/test_context_enforcer.py -v
"""
from src.nodes.context_enforcer import context_enforcer_node


def _make_chunk(doc: str, score: float, content_len: int = 100) -> dict:
    """Create a fake chunk with controllable content length."""
    return {
        "content": "x" * content_len,
        "source_doc": doc,
        "section": "chunk_0",
        "department": "HR",
        "score": score,
    }


def _make_state(chunks=None, memory=None, history=None) -> dict:
    return {
        "query": "test query",
        "user_id": "test",
        "retrieved_chunks": chunks or [],
        "memory_entries": memory or [],
        "conversation_history": history or [],
    }


def test_under_budget_passthrough():
    """When total tokens are under budget, nothing should be trimmed."""
    state = _make_state(
        chunks=[_make_chunk("doc1", 0.9, content_len=50)],
        memory=[{"fact": "user is in HR", "timestamp": "2025-01-01", "source_turn": 1}],
        history=[{"role": "user", "content": "hello"}],
    )
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is False
    assert len(result["trimmed_chunks"]) == 1
    assert len(result["trimmed_memory"]) == 1


def test_retrieval_trimming():
    """When retrieval exceeds budget, lowest-scoring chunks are dropped first."""
    chunks = [_make_chunk(f"doc{i}", score=i * 0.1, content_len=800) for i in range(10)]
    state = _make_state(chunks=chunks)
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is True
    assert len(result["trimmed_chunks"]) < 10
    # Remaining chunks should be the highest-scoring ones
    scores = [c["score"] for c in result["trimmed_chunks"]]
    assert scores == sorted(scores)


def test_conversation_trimming():
    """When conversation history exceeds budget, oldest turns are dropped."""
    history = [{"role": "user", "content": "message " * 100} for _ in range(20)]
    state = _make_state(history=history)
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is True
    assert len(result["trimmed_history"]) < 20


def test_memory_trimming():
    """When memory exceeds budget, oldest entries are dropped."""
    memory = [{"fact": f"fact {'x' * 200} {i}", "timestamp": f"2025-01-{i+1:02d}", "source_turn": i} for i in range(20)]
    state = _make_state(memory=memory)
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is True
    assert len(result["trimmed_memory"]) < 20


def test_budget_log_structure():
    """Budget log should always have the expected structure."""
    state = _make_state()
    result = context_enforcer_node(state)
    log = result["budget_log"]
    assert "zones" in log
    assert "total_before" in log
    assert "total_after" in log
    assert "enforced" in log
    assert "drop_details" in log
    assert len(log["zones"]) == 4
    zone_names = {z["zone_name"] for z in log["zones"]}
    assert zone_names == {"system_prompt", "retrieval", "memory", "conversation"}
```

---

### `tests/test_memory_store.py`

Unit tests for SQLite memory store CRUD operations.

```python
"""
test_memory_store — Unit tests for the SQLite memory store.
Run: python -m pytest tests/test_memory_store.py -v
"""
from src.stores import memory_store

TEST_USER = "test_user_unit"


def setup_function():
    """Clear test user data before each test."""
    memory_store.clear(TEST_USER)


def test_write_and_read():
    """Write facts and read them back."""
    memory_store.write(TEST_USER, ["fact one", "fact two"], turn=1)
    entries = memory_store.read(TEST_USER)
    assert len(entries) == 2
    facts = {e["fact"] for e in entries}
    assert "fact one" in facts
    assert "fact two" in facts


def test_read_empty():
    """Reading a non-existent user returns empty list."""
    entries = memory_store.read("nonexistent_user_xyz")
    assert entries == []


def test_write_empty_list():
    """Writing empty list should be a no-op."""
    count = memory_store.write(TEST_USER, [], turn=1)
    assert count == 0


def test_prune():
    """Prune should remove oldest entries beyond the cap."""
    for i in range(10):
        memory_store.write(TEST_USER, [f"fact {i}"], turn=i)
    deleted = memory_store.prune(TEST_USER, max_entries=3)
    assert deleted == 7
    remaining = memory_store.read(TEST_USER)
    assert len(remaining) == 3


def test_clear():
    """Clear should remove all entries for a user."""
    memory_store.write(TEST_USER, ["a", "b", "c"], turn=1)
    deleted = memory_store.clear(TEST_USER)
    assert deleted == 3
    entries = memory_store.read(TEST_USER)
    assert entries == []


def test_ordering():
    """Entries should be returned most recent first."""
    memory_store.write(TEST_USER, ["old fact"], turn=1)
    memory_store.write(TEST_USER, ["new fact"], turn=2)
    entries = memory_store.read(TEST_USER)
    assert entries[0]["fact"] == "new fact"
```

---

### `tests/test_router.py`

Lightweight smoke tests for the router node. These make real API calls so they need a valid `GOOGLE_API_KEY`.

```python
"""
test_router — Smoke tests for the router intent classifier.
These call the real Gemini API — requires GOOGLE_API_KEY in .env.
Run: python -m pytest tests/test_router.py -v
"""
import pytest
from src.nodes.router import router_node


def _make_state(query: str, history: list | None = None) -> dict:
    return {"query": query, "conversation_history": history or [], "user_id": "test"}


@pytest.mark.parametrize("query,expected_intent", [
    ("What is the leave policy?", "policy_lookup"),
    ("How many sick days do I get?", "policy_lookup"),
    ("What's the weather like?", "out_of_scope"),
    ("Tell me a joke", "out_of_scope"),
])
def test_basic_routing(query, expected_intent):
    """Router should classify obvious queries correctly."""
    result = router_node(_make_state(query))
    assert result["intent"] == expected_intent


def test_router_returns_required_keys():
    """Router output must contain intent, router_reasoning, retrieval_metadata_filter."""
    result = router_node(_make_state("What is the expense policy?"))
    assert "intent" in result
    assert "router_reasoning" in result
    assert "retrieval_metadata_filter" in result


def test_clarification_intent():
    """Router should detect clarification when conversation history exists."""
    history = [
        {"role": "user", "content": "What is the travel policy?"},
        {"role": "assistant", "content": "The travel policy covers booking class and per diem rates."},
    ]
    result = router_node(_make_state("Can you explain that in more detail?", history))
    assert result["intent"] == "clarification"
```

> [!WARNING]
> `test_router.py` makes real Gemini API calls. These tests will fail without a valid `GOOGLE_API_KEY` and will consume quota. Run them intentionally, not in CI.

---

## Changes from Original Spec

| Area | Spec | Changed To | Why |
|---|---|---|---|
| Phoenix setup | `px.launch_app()` only | Full OTEL tracer provider + `LangChainInstrumentor` | Needed for proper span export to Phoenix |
| Tracing init | No guard | Singleton `_initialized` flag | Prevents double-initialization if called multiple times |
| Test suite scoring | Human-scored answer correctness | `None` (deferred) + auto-scored router/source/hallucination | Phase 1 design — human scoring on first pass, LLM-as-judge in Phase 2 |
| Hallucination detection | Not specified | String matching for IDK phrases on trap questions | Simple, deterministic, catches the whistleblower trap and similar |
| Test runner | "Plain Python script" | Structured with `compute_aggregate`, `save_results`, `print_scorecard` | More useful output without adding framework overhead |
