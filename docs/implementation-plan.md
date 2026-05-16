# Context-Engineered RAG Agent — Implementation Plan

## Goal

Build a production-grade policy assistant agent over 20 synthetic company documents. The agent uses LangGraph for orchestration, enforces a strict 5000-token context budget across 4 zones, stores user memory in SQLite, retrieves policy chunks from Pinecone, and is fully instrumented with Arize Phoenix for observability. A 30-question stress-test suite validates correctness.

This plan is **architecture-level** — each module listed here will get its own detailed code reference document during execution.

---

## Project Structure

```
context-engineered-rag/
├── pyproject.toml                   # deps, project metadata
├── .env.example                     # required env vars template
├── README.md
│
├── corpus/                          # Phase 0 — generated offline
│   ├── documents/                   # 20 .txt policy files
│   └── test_suite.json              # 30 questions + ground truth
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py                    # all constants, budget caps, model names
│   │
│   ├── models/                      # Pydantic models for type safety
│   │   ├── __init__.py
│   │   ├── state.py                 # LangGraph AgentState TypedDict
│   │   └── schemas.py               # RouterOutput, MemoryEntry, TestQuestion, etc.
│   │
│   ├── stores/                      # data layer
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Pinecone: ingest, query, metadata filter
│   │   └── memory_store.py          # SQLite: read, write, prune by user
│   │
│   ├── nodes/                       # LangGraph node functions (1 file = 1 node)
│   │   ├── __init__.py
│   │   ├── router.py                # intent classification → 4 intents
│   │   ├── retriever.py             # semantic search against Pinecone
│   │   ├── memory_reader.py         # fetch user facts from SQLite
│   │   ├── context_enforcer.py      # token counting + trimming logic
│   │   ├── synthesizer.py           # main generation call (Gemini Pro)
│   │   ├── memory_writer.py         # extract & persist new user facts
│   │   └── out_of_scope.py          # fixed refusal response
│   │
│   ├── graph.py                     # assembles StateGraph, wires edges
│   ├── tracing.py                   # Phoenix + OTEL bootstrap
│   └── token_utils.py               # tiktoken-based counting helpers
│
├── scripts/
│   ├── generate_corpus.py           # LLM-assisted doc generation
│   ├── ingest_documents.py          # chunk + embed + upsert into Pinecone
│   └── run_test_suite.py            # execute 30 questions, produce scorecard
│
└── tests/                           # unit tests for isolated components
    ├── test_context_enforcer.py
    ├── test_memory_store.py
    └── test_router.py
```

---

## Dependencies

```
langgraph >= 0.4
langchain-google-genai            # Gemini Flash + Pro via API key (simpler than Vertex for dev)
langchain-pinecone                # Pinecone vector store integration
pinecone                          # Pinecone SDK (free Starter tier)
arize-phoenix
openinference-instrumentation-langchain
opentelemetry-sdk
opentelemetry-exporter-otlp
tiktoken                          # fast token counting
pydantic >= 2.0
python-dotenv
```

> [!IMPORTANT]
> The plan doc mentions **Vertex AI**, but for local dev simplicity we can use `langchain-google-genai` with a Gemini API key instead. Vertex is a one-line swap (`langchain-google-vertexai`) when deploying. **Confirm which approach you prefer.**

---

## Module-by-Module Architecture

### 1. `src/config.py` — Constants & Budget Caps

Single source of truth for every tunable parameter:

| Constant | Value | Notes |
|---|---|---|
| `SYSTEM_PROMPT_BUDGET` | 500 tokens | Fixed. Never grows. |
| `MEMORY_BUDGET` | 500 tokens | Oldest entries dropped first |
| `RETRIEVAL_BUDGET` | 3000 tokens | Lowest-relevance chunks dropped first |
| `CONVERSATION_BUDGET` | 1000 tokens | Older turns summarized when near cap |
| `TOTAL_BUDGET` | 5000 tokens | Hard ceiling |
| `RETRIEVAL_TOP_K` | 8 | Over-fetch, then trim to budget |
| `CHUNK_SIZE` | 512 tokens | For document chunking |
| `CHUNK_OVERLAP` | 50 tokens | |
| `ROUTER_MODEL` | `gemini-2.0-flash` | Fast/cheap |
| `SYNTHESIZER_MODEL` | `gemini-2.5-pro` | Quality |
| `MEMORY_WRITER_MODEL` | `gemini-2.0-flash` | |
| `EMBEDDING_MODEL` | `text-embedding-004` | |

---

### 2. `src/models/state.py` — LangGraph State

The central `TypedDict` that flows through every node:

```python
class AgentState(TypedDict):
    # Input
    user_id: str
    query: str
    conversation_history: list[dict]      # [{role, content}]

    # Router output
    intent: str                            # policy_lookup | clarification | memory_recall | out_of_scope
    router_reasoning: str                  # logged to trace

    # Retrieval output
    retrieved_chunks: list[dict]           # [{content, source_doc, section, score}]
    retrieval_metadata_filter: str | None  # department filter inferred by router

    # Memory output
    memory_entries: list[dict]             # [{fact, timestamp, source_turn}]

    # Budget enforcement output
    trimmed_chunks: list[dict]             # post-enforcement retrieval
    trimmed_history: list[dict]            # post-enforcement conversation
    trimmed_memory: list[dict]             # post-enforcement memory
    budget_log: dict                       # {zone: token_count, dropped: [...]}

    # Synthesis output
    response: str
    cited_sources: list[str]

    # Memory write output
    new_memory_entries: list[dict]
```

Every node reads what it needs, writes its output fields. LangGraph merges state automatically.

---

### 3. `src/nodes/router.py` — Intent Classification

- **Model**: Gemini Flash
- **Input**: `query` + `conversation_history` (last 3 turns only — cheap)
- **Output**: Structured JSON → `intent`, `router_reasoning`, optional `retrieval_metadata_filter`
- **Prompt**: Tight classification prompt with 4 intent definitions and 2-shot examples. Uses `response_mime_type: application/json` for structured output.
- **Key detail**: For `policy_lookup`, the router also infers a department tag (`HR`, `IT`, `Finance`, `Legal`, `Operations`) when the query makes it obvious. This tag feeds into Pinecone metadata filtering before semantic search.

---

### 4. `src/nodes/retriever.py` — Semantic Search

- **Only runs for** `policy_lookup` intent
- Queries Pinecone with the user's query embedding
- If `retrieval_metadata_filter` is set by the router, applies it as a `filter` dict on the `department` metadata field (e.g. `{"department": {"$eq": "Finance"}}`)
- Returns top-K chunks (K=8, over-fetched intentionally) with relevance scores
- **No LLM call** — pure vector similarity

---

### 5. `src/nodes/memory_reader.py` — Fetch User Facts

- Queries SQLite for all entries matching `user_id`, ordered by recency
- Runs for **all intents except `out_of_scope`**
- Returns raw entries — budget enforcement trims later
- **No LLM call**

---

### 6. `src/nodes/context_enforcer.py` — Budget Enforcement

The core context engineering node. **No LLM call — pure deterministic logic.**

Algorithm:
1. Count tokens in each zone using `tiktoken`
2. If total ≤ `TOTAL_BUDGET` → pass through unchanged
3. If over budget:
   - **Step A**: Sort `retrieved_chunks` by score ascending. Drop lowest-scoring chunks until retrieval zone ≤ `RETRIEVAL_BUDGET`
   - **Step B**: If still over, truncate `conversation_history` to most recent N turns that fit within `CONVERSATION_BUDGET`
   - **Step C**: If still over, drop oldest `memory_entries` until within `MEMORY_BUDGET`
4. Log every drop action into `budget_log` with: what was dropped, why, token counts before/after

The `budget_log` is the key observability artifact for this node — Phoenix traces will display it.

---

### 7. `src/nodes/synthesizer.py` — Main Generation

- **Model**: Gemini Pro
- **Input**: Assembled prompt from all 4 zones (system + trimmed memory + trimmed retrieval + trimmed history)
- **System prompt** includes: agent identity, citation rules ("always cite source document name"), behavioral constraints, and an explicit instruction to say "I don't have that information" rather than guessing
- **Output**: `response` text + `cited_sources` list
- This is the **only expensive LLM call** in the graph

---

### 8. `src/nodes/memory_writer.py` — Extract & Persist Facts

- **Model**: Gemini Flash
- **Runs after** synthesis
- Short extraction prompt: "Given this exchange, extract any user-specific facts worth remembering (role, preferences, open questions). Return JSON array or empty array."
- Writes new entries to SQLite with timestamp and source turn
- **No-op** for `out_of_scope` intent

---

### 9. `src/nodes/out_of_scope.py` — Fixed Refusal

- Returns a hardcoded polite refusal: "I'm a policy assistant for Meridian Technologies. I can help you with questions about company policies..."
- **No LLM call, no retrieval, no memory**
- Sets `response` in state and routes to END

---

### 10. `src/graph.py` — Graph Assembly

The LangGraph wiring:

```
START
  │
  ▼
[router]
  │
  ├── intent == "out_of_scope" ──► [out_of_scope] ──► END
  │
  ├── intent == "policy_lookup" ──► [retriever] ──┐
  │                                                 │ (parallel)
  │                                [memory_reader] ──┘
  │                                                 │
  │                                                 ▼
  │                                     [context_enforcer]
  │                                                 │
  │                                                 ▼
  │                                          [synthesizer]
  │                                                 │
  │                                                 ▼
  │                                        [memory_writer] ──► END
  │
  ├── intent == "clarification" ──► [memory_reader]
  │                                        │
  │                                        ▼
  │                               [context_enforcer]
  │                                        │
  │                                        ▼
  │                                  [synthesizer]
  │                                        │
  │                                        ▼
  │                                [memory_writer] ──► END
  │
  └── intent == "memory_recall" ──► [memory_reader]
                                           │
                                           ▼
                                  [context_enforcer]
                                           │
                                           ▼
                                     [synthesizer]
                                           │
                                           ▼
                                   [memory_writer] ──► END
```

Key implementation detail: `policy_lookup` is the only intent that runs the retriever. The other two non-OOS intents still pass through budget enforcement and synthesis — they just have an empty `retrieved_chunks` list.

> [!NOTE]
> LangGraph doesn't have native "parallel node" syntax but you can achieve parallel execution of `retriever` and `memory_reader` by having the router's conditional edge point to both, then using a "join" node that waits for both. Alternatively, a simpler approach: run them sequentially (retriever → memory_reader) since both are fast (no LLM calls). **Recommend sequential for Phase 1 simplicity.**

---

### 11. `src/stores/vector_store.py` — Pinecone

Requires `PINECONE_API_KEY` in `.env`. Uses the free Starter tier (sufficient for ~80 chunks).

Two entry points:
- **`ingest(docs_dir)`**: Reads .txt files, chunks with `RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)`, embeds with `text-embedding-004`, upserts into a Pinecone index. Each vector stores metadata: `{source_doc, section, department, text}`. Index is created on first run if it doesn't exist (dimension=768 for `text-embedding-004`, metric=cosine).
- **`query(text, k, metadata_filter)`**: Embeds query → Pinecone similarity search with optional metadata filter. Returns `[{content, source_doc, section, score}]`.

> [!NOTE]
> Pinecone stores metadata alongside vectors but **not** the raw chunk text by default. We store `text` as a metadata field so we can retrieve it without a separate lookup. This is standard practice for small corpora.

---

### 12. `src/stores/memory_store.py` — SQLite

Single table schema:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    timestamp TEXT NOT NULL,  -- ISO 8601
    source_turn INTEGER NOT NULL
);
CREATE INDEX idx_user_id ON memories(user_id);
```

Methods:
- `read(user_id) → list[MemoryEntry]` — all entries for user, ordered by timestamp DESC
- `write(user_id, facts: list[str], turn: int)` — batch insert
- `prune(user_id, max_entries)` — delete oldest beyond cap (safety valve)

---

### 13. `src/tracing.py` — Phoenix + OTEL Bootstrap

```python
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

def init_tracing():
    px.launch_app()  # starts Phoenix on localhost:6006
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter("http://127.0.0.1:6006/v1/traces"))
    )
    trace_api.set_tracer_provider(tracer_provider)
    LangChainInstrumentor().instrument()  # auto-instruments LangGraph
```

Called once at app startup. Every LangGraph node execution becomes a span in Phoenix automatically. The `budget_log` from the enforcer node will appear as span attributes.

---

### 14. `src/token_utils.py` — Token Counting

- Uses `tiktoken` with the `cl100k_base` encoding (close enough to Gemini tokenization for budget purposes)
- `count_tokens(text: str) → int`
- `count_messages(messages: list[dict]) → int`
- `truncate_to_budget(text: str, budget: int) → str`

> [!NOTE]
> Gemini's actual tokenizer differs slightly from tiktoken. For Phase 1, tiktoken is a good-enough proxy (within ~5%). If precision matters later, swap to `google.generativeai.count_tokens()` which calls the actual model tokenizer.

---

## Corpus Generation Strategy

### Documents (`scripts/generate_corpus.py`)

- Use Gemini Pro to generate each of the 20 documents based on the specifications in the plan (department, topic, edge conditions)
- Each document: 600–900 words, plain .txt, prefixed with metadata header:
  ```
  DOCUMENT: Leave Policy
  DEPARTMENT: HR
  VERSION: 2.1
  LAST UPDATED: 2025-01-15
  ---
  [body text]
  ```
- The 6 edge-condition documents get explicit contradictions/traps baked in during generation

### Test Suite (`corpus/test_suite.json`)

```json
[
  {
    "id": 1,
    "question": "How many days of casual leave can I take per year?",
    "category": "straightforward",
    "expected_intent": "policy_lookup",
    "expected_sources": ["Leave Policy"],
    "expected_answer": "Employees are entitled to 12 days of casual leave per calendar year.",
    "triggers_budget_enforcement": false,
    "correct_answer_is_idk": false
  },
  ...
]
```

---

## Build Phases

### Phase A: Corpus (no code)
Generate 20 documents + 30 test questions with ground truth. Review manually for quality.

### Phase B: Infrastructure
1. `pyproject.toml` — lock all deps
2. `src/config.py` — all constants
3. `src/tracing.py` — Phoenix bootstrap, verify traces appear
4. `src/token_utils.py` — counting helpers
5. `src/models/` — state and schemas
6. `src/stores/vector_store.py` — ingest + query
7. `src/stores/memory_store.py` — SQLite CRUD
8. `scripts/ingest_documents.py` — embed + upsert corpus into Pinecone

### Phase C: Graph (incremental)
1. `router` + `synthesizer` + `out_of_scope` only. Wire graph. Verify traces.
2. Add `retriever`. Verify chunk scores appear in traces.
3. Add `context_enforcer`. Verify enforcement triggers on long contexts.
4. Add `memory_reader` + `memory_writer`. Verify memory round-trips.

### Phase D: Validation
1. Run `scripts/run_test_suite.py` against full graph
2. Score first pass manually (answer correctness)
3. Produce 3×30 scorecard

### Phase E: Tuning
1. Analyze Phoenix traces for failure patterns
2. Fix prompts, chunking, or routing issues
3. Re-run suite

### Phase F: Final Scorecard
Target: Router ≥ 90%, Source ≥ 85%, Hallucination = 0% on traps.

---

## Open Questions

> [!IMPORTANT]
> **Gemini API Key vs Vertex AI**: The plan doc says Vertex AI. For local dev, a plain Gemini API key (`langchain-google-genai`) is simpler and avoids GCP project setup. Vertex is a one-line swap later. Which do you want for Phase 1?

> [!IMPORTANT]
> **Parallel vs Sequential retriever/memory_reader**: Running them in parallel is architecturally correct but adds LangGraph wiring complexity (fan-out/fan-in). Running them sequentially is simpler and equally fast since neither calls an LLM. Recommendation: **sequential for Phase 1**. Confirm?

> [!NOTE]
> **Tiktoken vs Gemini tokenizer**: Tiktoken (`cl100k_base`) is ~5% off from Gemini's actual tokenization. Good enough for budget enforcement in Phase 1, and it's fast + local. The alternative is calling `genai.count_tokens()` which is an API call per count. Recommend tiktoken.

---

## What Comes Next (after this plan)

Individual **code reference documents** for each module group:
1. **Corpus Code Ref** — `generate_corpus.py` + document specs + test suite schema
2. **Stores Code Ref** — `vector_store.py` (Pinecone) + `memory_store.py` (SQLite) + `ingest_documents.py`
3. **Nodes Code Ref** — all 7 node files + prompts
4. **Graph Code Ref** — `graph.py` + `state.py` + `tracing.py`
5. **Validation Code Ref** — `run_test_suite.py` + scoring logic

Each code ref will contain complete, runnable code — no stubs, no placeholders.
