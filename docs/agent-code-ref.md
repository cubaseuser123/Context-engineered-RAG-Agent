# Agent Code Reference

> [!IMPORTANT]
> Build files in the exact order listed. Each module only imports from modules above it.

## Project Init

```powershell
mkdir context-engineered-rag; cd context-engineered-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Folder Structure

```
context-engineered-rag/
├── pyproject.toml                   # deps, project metadata
├── .env.example                     # required env vars template
├── .gitignore
│
├── corpus/                          # generated offline (Corpus Code Ref)
│   ├── documents/                   # 20 .txt policy files
│   └── test_suite.json              # 30 questions + ground truth
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py                    # all constants, budget caps, model names
│   ├── token_utils.py               # tiktoken counting (Context Engineering Ref)
│   ├── tracing.py                   # Phoenix + OTEL bootstrap (Observability Ref)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py               # RouterOutput, MemoryEntry, BudgetLog, etc.
│   │   └── state.py                 # LangGraph AgentState TypedDict
│   │
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # Pinecone: ingest, query, metadata filter
│   │   └── memory_store.py          # SQLite: read, write, prune by user
│   │
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── router.py                # intent classification → 4 intents
│   │   ├── retriever.py             # semantic search against Pinecone
│   │   ├── memory_reader.py         # fetch user facts from SQLite
│   │   ├── context_enforcer.py      # budget enforcement (Context Engineering Ref)
│   │   ├── synthesizer.py           # main generation call (Gemini Pro)
│   │   ├── memory_writer.py         # extract & persist new user facts
│   │   └── out_of_scope.py          # fixed refusal response
│   │
│   └── graph.py                     # assembles StateGraph, wires edges
│
├── scripts/
│   ├── __init__.py
│   ├── generate_corpus.py           # LLM-assisted doc generation (Corpus Code Ref)
│   ├── ingest_documents.py          # chunk + embed + upsert (Context Engineering Ref)
│   └── run_test_suite.py            # 30-question validation (Observability Ref)
│
├── tests/
│   ├── test_context_enforcer.py     # (Observability Ref)
│   ├── test_memory_store.py         # (Observability Ref)
│   └── test_router.py              # (Observability Ref)
│
└── results/                         # auto-created by test suite
    └── scorecard_*.json
```

## Dependencies

### `pyproject.toml`

```toml
[project]
name = "context-engineered-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.4",
    "langchain-google-genai",
    "langchain-pinecone",
    "pinecone",
    "arize-phoenix",
    "openinference-instrumentation-langchain",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "tiktoken",
    "pydantic>=2.0",
    "python-dotenv",
    "google-generativeai",
]
```

```powershell
pip install -e .
```

## Environment

### `.env.example`

```env
GOOGLE_API_KEY=your-gemini-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=meridian-policies
```

---

## Build Order: config → schemas → state → memory_store → vector_store

---

### `src/__init__.py`

```python
"""src — Root package for the context-engineered RAG agent."""
```

---

### `src/config.py`

Every constant, budget cap, and env var lives here. No other module touches `os.environ`.

```python
"""
config — Single source of truth for constants, budget caps, model names.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]

# --- Pinecone ---
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "meridian-policies")
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"
EMBEDDING_DIMENSION: int = 768  # text-embedding-004

# --- Models ---
ROUTER_MODEL: str = "gemini-2.0-flash"
SYNTHESIZER_MODEL: str = "gemini-2.5-pro"
MEMORY_WRITER_MODEL: str = "gemini-2.0-flash"
EMBEDDING_MODEL: str = "models/text-embedding-004"

# --- Context Budget (tokens) ---
SYSTEM_PROMPT_BUDGET: int = 500
MEMORY_BUDGET: int = 500
RETRIEVAL_BUDGET: int = 3000
CONVERSATION_BUDGET: int = 1000
TOTAL_BUDGET: int = 5000

# --- Retrieval ---
RETRIEVAL_TOP_K: int = 8
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = PROJECT_ROOT / "corpus" / "documents"
TEST_SUITE_PATH: Path = PROJECT_ROOT / "corpus" / "test_suite.json"
MEMORY_DB_PATH: Path = PROJECT_ROOT / "memory.db"

# --- Phoenix ---
PHOENIX_ENDPOINT: str = "http://127.0.0.1:6006/v1/traces"
PHOENIX_PROJECT_NAME: str = "context-engineered-rag"

# --- Valid Values ---
VALID_INTENTS: list[str] = ["policy_lookup", "clarification", "memory_recall", "out_of_scope"]
VALID_DEPARTMENTS: list[str] = ["HR", "IT", "Finance", "Legal", "Operations"]
```

---

### `src/models/__init__.py`

```python
"""models — Pydantic schemas and LangGraph state definitions."""
```

### `src/models/schemas.py`

Pydantic models for structured data. No local imports.

```python
"""
schemas — Pydantic models for structured data throughout the agent.
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    intent: str = Field(description="One of: policy_lookup, clarification, memory_recall, out_of_scope")
    reasoning: str = Field(description="Why this intent was selected")
    department_filter: str | None = Field(default=None, description="Department for metadata filtering")


class MemoryEntry(BaseModel):
    fact: str
    timestamp: str
    source_turn: int


class RetrievedChunk(BaseModel):
    content: str
    source_doc: str
    section: str
    department: str
    score: float


class BudgetZoneReport(BaseModel):
    zone_name: str
    token_count: int
    budget: int
    items_dropped: int = 0


class BudgetLog(BaseModel):
    zones: list[BudgetZoneReport] = Field(default_factory=list)
    total_before: int = 0
    total_after: int = 0
    enforced: bool = False
    drop_details: list[str] = Field(default_factory=list)


class TestQuestion(BaseModel):
    id: int
    question: str
    category: str
    expected_intent: str
    expected_sources: list[str]
    expected_answer: str
    triggers_budget_enforcement: bool
    correct_answer_is_idk: bool


class ScorecardEntry(BaseModel):
    question_id: int
    router_correct: bool
    source_correct: bool
    answer_correct: bool | None = None
    hallucinated: bool = False
```

---

### `src/models/state.py`

The LangGraph state contract between all nodes.

```python
"""
state — LangGraph AgentState TypedDict.
Every node reads from and writes to this shared state.
"""
from __future__ import annotations
from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Input (set by caller)
    user_id: str
    query: str
    conversation_history: list[dict[str, str]]
    turn_number: int

    # Router output
    intent: str
    router_reasoning: str
    retrieval_metadata_filter: str | None

    # Retrieval output
    retrieved_chunks: list[dict[str, Any]]

    # Memory reader output
    memory_entries: list[dict[str, Any]]

    # Context enforcer output
    trimmed_chunks: list[dict[str, Any]]
    trimmed_history: list[dict[str, str]]
    trimmed_memory: list[dict[str, Any]]
    budget_log: dict[str, Any]

    # Synthesizer output
    response: str
    cited_sources: list[str]

    # Memory writer output
    new_memory_entries: list[dict[str, Any]]
```

> [!NOTE]
> `total=False` makes all fields optional — nodes only write their outputs. Without this, LangGraph would require every field at graph invocation.

---

### `src/stores/__init__.py`

```python
"""stores — Data access layer for Pinecone and SQLite."""
```

### `src/stores/memory_store.py`

SQLite user memory. Simple CRUD, no embeddings.

```python
"""
memory_store — SQLite-backed user memory store.
Stores user-specific facts. Read by memory_reader, written by memory_writer.
"""
import logging
import sqlite3
from datetime import datetime, timezone
from src.config import MEMORY_DB_PATH

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source_turn INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    return conn


def read(user_id: str) -> list[dict]:
    """Fetch all entries for a user, most recent first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT fact, timestamp, source_turn FROM memories WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
        ).fetchall()
        logger.info("Memory read user='%s': %d entries", user_id, len(rows))
        return [dict(r) for r in rows]
    finally:
        conn.close()


def write(user_id: str, facts: list[str], turn: int) -> int:
    """Batch insert new facts. Returns count written."""
    if not facts:
        return 0
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.executemany(
            "INSERT INTO memories (user_id, fact, timestamp, source_turn) VALUES (?, ?, ?, ?)",
            [(user_id, f, now, turn) for f in facts],
        )
        conn.commit()
        logger.info("Memory write user='%s': %d entries", user_id, len(facts))
        return len(facts)
    finally:
        conn.close()


def prune(user_id: str, max_entries: int = 20) -> int:
    """Delete oldest entries beyond cap. Returns count deleted."""
    conn = _get_conn()
    try:
        count = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)).fetchone()[0]
        if count <= max_entries:
            return 0
        to_del = count - max_entries
        conn.execute(
            "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE user_id = ? ORDER BY timestamp ASC LIMIT ?)",
            (user_id, to_del),
        )
        conn.commit()
        logger.info("Memory prune user='%s': %d deleted", user_id, to_del)
        return to_del
    finally:
        conn.close()


def clear(user_id: str) -> int:
    """Delete all entries for a user. Testing utility."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
```

---

### `src/stores/vector_store.py`

Pinecone vector store. Index creation, ingestion, and similarity search.

```python
"""
vector_store — Pinecone vector store for policy document chunks.
Handles index creation, ingestion (chunk + embed + upsert), and search.
"""
import hashlib
import logging
from pathlib import Path

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL,
    GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_CLOUD,
    PINECONE_INDEX_NAME, PINECONE_REGION, RETRIEVAL_TOP_K,
)

logger = logging.getLogger(__name__)

_pc: Pinecone | None = None
_index = None
genai.configure(api_key=GOOGLE_API_KEY)


def _get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def _get_index():
    global _index
    if _index is not None:
        return _index
    pc = _get_pc()
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s'", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def _embed_texts(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(model=EMBEDDING_MODEL, content=texts, task_type="retrieval_document")
    return result["embedding"]


def _embed_query(text: str) -> list[float]:
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_query")
    return result["embedding"]


def _chunk_id(doc: str, idx: int) -> str:
    return hashlib.md5(f"{doc}::chunk_{idx}".encode()).hexdigest()


def _parse_doc(path: Path) -> dict:
    """Parse policy doc with metadata header (DOCUMENT/DEPARTMENT/VERSION lines above ---)."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 1)
    meta = {}
    if len(parts) == 2:
        for line in parts[0].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip().lower().replace(" ", "_")] = v.strip()
        body = parts[1].strip()
    else:
        body = text.strip()
    return {"source_doc": meta.get("document", path.stem), "department": meta.get("department", "Unknown"), "body": body}


def _chunk_text(text: str) -> list[str]:
    """Split text into chunks. Uses ~4 chars/token heuristic."""
    char_chunk = CHUNK_SIZE * 4
    char_overlap = CHUNK_OVERLAP * 4
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + char_chunk].strip()
        if chunk:
            chunks.append(chunk)
        start += char_chunk - char_overlap
    return chunks


def ingest(docs_dir: Path) -> int:
    """Ingest all .txt docs into Pinecone. Returns total chunks upserted."""
    index = _get_index()
    files = sorted(docs_dir.glob("*.txt"))
    total = 0
    for fp in files:
        doc = _parse_doc(fp)
        chunks = _chunk_text(doc["body"])
        if not chunks:
            continue
        embeddings = _embed_texts(chunks)
        vectors = [
            (_chunk_id(doc["source_doc"], i), emb, {"source_doc": doc["source_doc"], "department": doc["department"], "section": f"chunk_{i}", "text": ch})
            for i, (ch, emb) in enumerate(zip(chunks, embeddings))
        ]
        for b in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[b : b + 100], namespace="policies")
        total += len(chunks)
        logger.info("%s → %d chunks", doc["source_doc"], len(chunks))
    logger.info("Ingestion done: %d chunks", total)
    return total


def query(text: str, k: int = RETRIEVAL_TOP_K, metadata_filter: str | None = None) -> list[dict]:
    """Semantic search. Returns [{content, source_doc, section, department, score}]."""
    index = _get_index()
    emb = _embed_query(text)
    filt = {"department": {"$eq": metadata_filter}} if metadata_filter else None
    results = index.query(vector=emb, top_k=k, namespace="policies", include_metadata=True, filter=filt)
    chunks = []
    for m in results.matches:
        meta = m.metadata or {}
        chunks.append({"content": meta.get("text", ""), "source_doc": meta.get("source_doc", ""), "section": meta.get("section", ""), "department": meta.get("department", ""), "score": float(m.score)})
    logger.info("Query returned %d chunks (filter=%s)", len(chunks), metadata_filter)
    return chunks
```

> [!NOTE]
> `task_type` differs: `retrieval_document` for ingestion, `retrieval_query` for search. This is a Google embedding model requirement.

---

## Part 2: Nodes + Graph

> [!IMPORTANT]
> Build order continues from Part 1. Nodes depend on config, state, and stores. `graph.py` comes last — it imports all nodes. `context_enforcer.py` and `token_utils.py` are in the **Context Engineering Code Ref**.

**Build order**: router → retriever → memory_reader → synthesizer → memory_writer → out_of_scope → graph

---

### `src/nodes/__init__.py`

```python
"""nodes — LangGraph node functions. One file per node."""
```

---

### `src/nodes/router.py`

Intent classification node. Fast Gemini Flash call with structured JSON output. Classifies into 4 intents and optionally infers a department filter.

```python
"""
router — Intent classification node.
Takes: query + conversation_history. Returns: intent, reasoning, optional department filter.
Uses Gemini Flash for speed. This is a cheap call.
"""
import json
import logging

import google.generativeai as genai

from src.config import GOOGLE_API_KEY, ROUTER_MODEL, VALID_DEPARTMENTS, VALID_INTENTS
from src.models.state import AgentState

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

ROUTER_SYSTEM_PROMPT = """You are a query intent classifier for a company policy assistant at Meridian Technologies.

Classify the user's query into exactly one intent:
- "policy_lookup": User wants information from a company policy document.
- "clarification": User is following up or asking for clarification on something already discussed.
- "memory_recall": User is asking about something they previously told you (their role, preferences, etc).
- "out_of_scope": Query has nothing to do with company policy.

If the intent is "policy_lookup", also infer which department the query relates to if obvious:
HR, IT, Finance, Legal, Operations. Set department_filter to null if unclear.

Respond with ONLY valid JSON:
{"intent": "...", "reasoning": "...", "department_filter": "..." or null}

Examples:
User: "What is the travel reimbursement limit?"
{"intent": "policy_lookup", "reasoning": "User asks about travel reimbursement which falls under Finance policies", "department_filter": "Finance"}

User: "Can you explain that last point in more detail?"
{"intent": "clarification", "reasoning": "User references previous conversation with 'that last point'", "department_filter": null}

User: "What's the weather like today?"
{"intent": "out_of_scope", "reasoning": "Weather has nothing to do with company policy", "department_filter": null}
"""


def router_node(state: AgentState) -> dict:
    """Classify user query intent. Returns intent, reasoning, and optional department filter."""
    query = state.get("query", "")
    history = state.get("conversation_history", [])

    # Only pass last 3 turns to keep this call cheap
    recent_history = history[-3:] if len(history) > 3 else history
    history_text = ""
    for turn in recent_history:
        history_text += f"{turn.get('role', 'user')}: {turn.get('content', '')}\n"

    user_message = f"Conversation history:\n{history_text}\n\nCurrent query: {query}"

    model = genai.GenerativeModel(
        ROUTER_MODEL,
        system_instruction=ROUTER_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    response = model.generate_content(user_message)
    raw = response.text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Router returned invalid JSON: %s", raw)
        parsed = {"intent": "out_of_scope", "reasoning": "Failed to parse router output", "department_filter": None}

    intent = parsed.get("intent", "out_of_scope")
    if intent not in VALID_INTENTS:
        logger.warning("Router returned invalid intent '%s', defaulting to out_of_scope", intent)
        intent = "out_of_scope"

    dept = parsed.get("department_filter")
    if dept and dept not in VALID_DEPARTMENTS:
        dept = None

    logger.info("Router: intent=%s, dept=%s, reasoning=%s", intent, dept, parsed.get("reasoning", ""))

    return {
        "intent": intent,
        "router_reasoning": parsed.get("reasoning", ""),
        "retrieval_metadata_filter": dept,
    }
```

---

### `src/nodes/retriever.py`

Semantic search node. Only runs for `policy_lookup` intent. No LLM call — pure vector similarity via Pinecone.

```python
"""
retriever — Semantic search node against Pinecone.
Takes: query, retrieval_metadata_filter. Returns: retrieved_chunks.
Only activated for policy_lookup intent. No LLM call.
"""
import logging

from src.config import RETRIEVAL_TOP_K
from src.models.state import AgentState
from src.stores import vector_store

logger = logging.getLogger(__name__)


def retriever_node(state: AgentState) -> dict:
    """Query vector store for relevant policy chunks."""
    query = state.get("query", "")
    metadata_filter = state.get("retrieval_metadata_filter")

    chunks = vector_store.query(
        text=query,
        k=RETRIEVAL_TOP_K,
        metadata_filter=metadata_filter,
    )

    logger.info(
        "Retriever: %d chunks returned (filter=%s, top_score=%.3f)",
        len(chunks),
        metadata_filter,
        chunks[0]["score"] if chunks else 0.0,
    )

    return {"retrieved_chunks": chunks}
```

---

### `src/nodes/memory_reader.py`

Fetches user-specific facts from SQLite. Runs for all intents except `out_of_scope`. No LLM call.

```python
"""
memory_reader — Fetch user facts from SQLite memory store.
Takes: user_id. Returns: memory_entries.
Runs for all intents except out_of_scope. No LLM call.
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
    return {"memory_entries": entries}
```

---

### `src/nodes/synthesizer.py`

Main generation node. The only expensive LLM call (Gemini Pro). Assembles all 4 context zones into a single prompt and generates the response with source citations.

```python
"""
synthesizer — Main generation node using Gemini Pro.
Takes: trimmed context from all 4 zones. Returns: response + cited_sources.
This is the only expensive LLM call in the graph.
"""
import json
import logging

import google.generativeai as genai

from src.config import GOOGLE_API_KEY, SYNTHESIZER_MODEL
from src.models.state import AgentState

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are a policy assistant for Meridian Technologies, a mid-sized B2B SaaS company (~800 employees, HQ Pune).

RULES:
1. Answer ONLY based on the policy documents provided in the context below.
2. ALWAYS cite the source document name in your answer (e.g. "According to the Leave Policy...").
3. If the answer is not in the provided documents, say "I don't have that information in the available policy documents."
4. NEVER invent policy details, phone numbers, email addresses, or any information not explicitly in the documents.
5. If documents contain conflicting information, acknowledge the conflict and present both versions.
6. Be concise and direct. Employees want quick answers.

You will receive context in these sections:
- MEMORY: Facts about this user from previous conversations
- DOCUMENTS: Relevant policy document excerpts
- CONVERSATION: Recent exchange history
"""


def _build_context(state: AgentState) -> str:
    """Assemble the 4 context zones into a single user message."""
    parts = []

    # Memory zone
    memory = state.get("trimmed_memory", [])
    if memory:
        mem_text = "\n".join(f"- {m.get('fact', '')}" for m in memory)
        parts.append(f"=== MEMORY ===\n{mem_text}")

    # Retrieval zone
    chunks = state.get("trimmed_chunks", [])
    if chunks:
        doc_text = ""
        for c in chunks:
            doc_text += f"\n[Source: {c.get('source_doc', 'unknown')}]\n{c.get('content', '')}\n"
        parts.append(f"=== DOCUMENTS ==={doc_text}")

    # Conversation zone
    history = state.get("trimmed_history", [])
    if history:
        hist_text = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history)
        parts.append(f"=== CONVERSATION ===\n{hist_text}")

    # Current query
    parts.append(f"=== CURRENT QUESTION ===\n{state.get('query', '')}")

    return "\n\n".join(parts)


def synthesizer_node(state: AgentState) -> dict:
    """Generate the final response with source citations."""
    context = _build_context(state)

    model = genai.GenerativeModel(
        SYNTHESIZER_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.2),
    )

    response = model.generate_content(context)
    answer = response.text.strip()

    # Extract cited sources from the response text
    chunks = state.get("trimmed_chunks", [])
    source_docs = {c.get("source_doc", "") for c in chunks}
    cited = [s for s in source_docs if s and s.lower() in answer.lower()]

    # If no sources detected in text, include all provided sources
    if not cited and chunks:
        cited = list(source_docs)

    logger.info("Synthesizer: response=%d chars, cited=%s", len(answer), cited)

    return {
        "response": answer,
        "cited_sources": cited,
    }
```

---

### `src/nodes/memory_writer.py`

Extracts user-specific facts from the exchange and persists them. Uses Gemini Flash for cheap extraction.

```python
"""
memory_writer — Extract and persist user facts after synthesis.
Takes: query, response, user_id. Returns: new_memory_entries.
Uses Gemini Flash. No-op for out_of_scope intent.
"""
import json
import logging

import google.generativeai as genai

from src.config import GOOGLE_API_KEY, MEMORY_WRITER_MODEL
from src.models.state import AgentState
from src.stores import memory_store

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

EXTRACTION_PROMPT = """Analyze this conversation exchange and extract any user-specific facts worth remembering for future conversations.

Facts to extract:
- User's role or department
- User's preferences or constraints
- Specific topics the user is tracking
- Open questions the user mentioned

Return a JSON array of strings. Each string is one fact.
Return an empty array [] if nothing worth remembering.

User query: {query}
Assistant response: {response}

JSON array:"""


def memory_writer_node(state: AgentState) -> dict:
    """Extract and store new user facts from this exchange."""
    intent = state.get("intent", "")
    if intent == "out_of_scope":
        logger.info("Memory writer: skipped (out_of_scope)")
        return {"new_memory_entries": []}

    query = state.get("query", "")
    response = state.get("response", "")
    user_id = state.get("user_id", "default")
    turn = state.get("turn_number", 0)

    model = genai.GenerativeModel(
        MEMORY_WRITER_MODEL,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    prompt = EXTRACTION_PROMPT.format(query=query, response=response)
    result = model.generate_content(prompt)

    try:
        facts = json.loads(result.text.strip())
        if not isinstance(facts, list):
            facts = []
        facts = [f for f in facts if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        logger.warning("Memory writer: failed to parse JSON: %s", result.text)
        facts = []

    if facts:
        memory_store.write(user_id, facts, turn)
        memory_store.prune(user_id)

    logger.info("Memory writer: %d facts extracted and stored", len(facts))

    return {"new_memory_entries": [{"fact": f} for f in facts]}
```

---

### `src/nodes/out_of_scope.py`

Fixed polite refusal. No LLM call, no retrieval, no memory.

```python
"""
out_of_scope — Fixed refusal handler for out-of-scope queries.
No LLM call. Returns a hardcoded response.
"""
import logging

from src.models.state import AgentState

logger = logging.getLogger(__name__)

REFUSAL_RESPONSE = (
    "I'm a policy assistant for Meridian Technologies. I can help you with questions about "
    "company policies including HR, IT & Security, Finance, Legal & Compliance, and Operations. "
    "Could you rephrase your question to relate to a company policy?"
)


def out_of_scope_node(state: AgentState) -> dict:
    """Return a fixed polite refusal without any LLM call."""
    logger.info("Out of scope handler: returning fixed refusal")
    return {
        "response": REFUSAL_RESPONSE,
        "cited_sources": [],
    }
```

---

### `src/graph.py`

Assembles the full LangGraph StateGraph. Wires all nodes with conditional routing based on intent.

```python
"""
graph — LangGraph StateGraph assembly.
Wires all nodes with conditional edges based on router intent.
This is the entrypoint for running the agent.
"""
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.models.state import AgentState
from src.nodes.memory_reader import memory_reader_node
from src.nodes.memory_writer import memory_writer_node
from src.nodes.out_of_scope import out_of_scope_node
from src.nodes.retriever import retriever_node
from src.nodes.router import router_node
from src.nodes.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def _route_by_intent(state: AgentState) -> Literal["retriever", "memory_reader", "out_of_scope"]:
    """Conditional edge: route based on classified intent."""
    intent = state.get("intent", "out_of_scope")
    if intent == "policy_lookup":
        return "retriever"
    elif intent == "out_of_scope":
        return "out_of_scope"
    else:
        # clarification and memory_recall both go to memory_reader (skip retriever)
        return "memory_reader"


def build_graph() -> StateGraph:
    """Build and compile the agent graph."""

    graph = StateGraph(AgentState)

    # --- Add nodes ---
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("memory_reader", memory_reader_node)
    graph.add_node("context_enforcer", _get_enforcer_node())
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("memory_writer", memory_writer_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    # --- Wire edges ---

    # START → router
    graph.add_edge(START, "router")

    # router → conditional split
    graph.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "retriever": "retriever",
            "memory_reader": "memory_reader",
            "out_of_scope": "out_of_scope",
        },
    )

    # policy_lookup path: retriever → memory_reader → enforcer → synthesizer → memory_writer
    graph.add_edge("retriever", "memory_reader")

    # All non-OOS paths converge: memory_reader → enforcer → synthesizer → memory_writer → END
    graph.add_edge("memory_reader", "context_enforcer")
    graph.add_edge("context_enforcer", "synthesizer")
    graph.add_edge("synthesizer", "memory_writer")
    graph.add_edge("memory_writer", END)

    # OOS path: out_of_scope → END
    graph.add_edge("out_of_scope", END)

    return graph


def _get_enforcer_node():
    """Import context_enforcer lazily to avoid circular deps. Covered in Context Engineering Ref."""
    from src.nodes.context_enforcer import context_enforcer_node
    return context_enforcer_node


def compile_agent():
    """Build, compile, and return the runnable agent."""
    graph = build_graph()
    app = graph.compile()
    logger.info("Agent graph compiled successfully")
    return app


def run_query(app, query: str, user_id: str = "default", conversation_history: list | None = None, turn: int = 0) -> dict:
    """Convenience function to invoke the agent with a query."""
    state = {
        "query": query,
        "user_id": user_id,
        "conversation_history": conversation_history or [],
        "turn_number": turn,
    }
    result = app.invoke(state)
    return result
```

> [!NOTE]
> `context_enforcer` is imported lazily via `_get_enforcer_node()`. The full implementation is in the **Context Engineering Code Ref**. The graph will fail to compile until that file exists.

> [!NOTE]
> `policy_lookup` runs retriever → memory_reader sequentially (not parallel). Both are fast non-LLM calls, so the latency impact is negligible. This avoids LangGraph fan-out/fan-in complexity.

---

## Running the Agent (after all code refs are built)

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Set up .env with your keys
Copy-Item .env.example .env
# Edit .env with real keys

# 3. Ingest corpus (covered in Context Engineering Ref)
python -m scripts.ingest_documents

# 4. Run a single query
python -c "from src.graph import compile_agent, run_query; app = compile_agent(); print(run_query(app, 'What is the leave policy?')['response'])"
```

---

## Changes from Original Spec

| Area | Spec | Changed To | Why |
|---|---|---|---|
| Vector store | ChromaDB | Pinecone | User decision |
| Embedding calls | LangChain wrapper | Direct `google.generativeai` | Fewer abstractions, easier tracing |
| State TypedDict | `total=True` implied | `total=False` | Nodes write partial state |
| Chunking | `RecursiveCharacterTextSplitter` | Custom `_chunk_text` | Avoids extra dep for one function |
| Parallel retriever+memory | Fan-out/fan-in | Sequential (retriever → memory_reader) | Simpler wiring, same speed (no LLM calls) |
| Router structured output | Pydantic model parsing | `response_mime_type="application/json"` + manual parse | More reliable with Gemini, avoids LangChain output parser |
| Source citation extraction | Structured output from LLM | String matching against provided source names | Simpler, deterministic, no extra LLM call |
