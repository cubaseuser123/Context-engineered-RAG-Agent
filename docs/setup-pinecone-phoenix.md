# Setup: Pinecone + Arize Phoenix

---

## Part 1 — Pinecone

### Step 1 — Create a free account

Go to [https://app.pinecone.io](https://app.pinecone.io) → Sign up with Google or email.

The **Starter** plan is free and handles this project's ~80 chunks with no card required.

---

### Step 2 — Get your API key

1. Log in to the Pinecone console
2. Left sidebar → **API Keys**
3. Copy the default key (labelled `default`)

Paste it into your `.env`:

```env
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_INDEX_NAME=meridian-policies
```

> [!NOTE]
> The index (`meridian-policies`) does **not** need to be created manually. `vector_store.py` creates it on first run via `pc.create_index()`. Just set the name in `.env`.

---

### Step 3 — Verify the connection

With your venv active, run:

```powershell
python -c "from src.stores.vector_store import _get_pc; pc = _get_pc(); print([i.name for i in pc.list_indexes()])"
```

**Expected output**: `[]` (empty list — no indexes yet, but connection works)

If you see an `AuthenticationError`, your API key is wrong or `.env` isn't loading.

---

### Step 4 — Run ingestion (after corpus is ready)

```powershell
python -m scripts.ingest_documents
```

After ingestion, verify the index exists:

```powershell
python -c "from src.stores.vector_store import _get_pc; pc = _get_pc(); print([i.name for i in pc.list_indexes()])"
```

**Expected**: `['meridian-policies']`

You can also check in the Pinecone console: **Indexes** → click `meridian-policies` → see vector count.

---

## Part 2 — Arize Phoenix

### Step 1 — No account needed

Phoenix runs entirely **locally**. No sign-up, no API key. It's just a Python process.

It is already in your dependencies (`arize-phoenix`). Confirm it installed:

```powershell
python -c "import phoenix; print(phoenix.__version__)"
```

---

### Step 2 — How it starts

Phoenix is launched inside your code via `src/tracing.py`:

```python
px.launch_app()  # starts the Phoenix UI server on localhost:6006
```

`init_tracing()` is called at the top of `run_test_suite.py` and can be called from any entrypoint. You don't run Phoenix separately.

---

### Step 3 — Start it manually to test

```powershell
python -c "import phoenix as px; px.launch_app(); input('Phoenix running — press Enter to stop')"
```

Then open your browser: [http://localhost:6006](http://localhost:6006)

You should see the Phoenix UI with an empty **Traces** table.

---

### Step 4 — Run the agent and see traces appear

```powershell
python -c "
from src.tracing import init_tracing
from src.graph import compile_agent, run_query
init_tracing()
app = compile_agent()
result = run_query(app, 'What is the leave policy?')
print(result['response'])
input('Check http://localhost:6006 — press Enter to stop')
"
```

Refresh Phoenix → you should see **1 trace** with the full node tree (router → retriever → memory_reader → context_enforcer → synthesizer → memory_writer).

Click any node to see its inputs, outputs, and token counts.

---

### What to look for in the UI

| Span | What to check |
|---|---|
| `router` | `intent` field in outputs — did it classify correctly? |
| `retriever` | `retrieved_chunks` — how many chunks? What scores? |
| `context_enforcer` | `budget_log.enforced` — did it trim anything? `drop_details` shows what was dropped |
| `synthesizer` | Full prompt input (all 4 zones assembled) + response |
| `memory_writer` | `new_memory_entries` — what facts were extracted? |

---

## Quick Reference

| Thing | Where |
|---|---|
| Pinecone console | https://app.pinecone.io |
| Phoenix UI | http://localhost:6006 (local, only when running) |
| API key env var | `PINECONE_API_KEY` in `.env` |
| Phoenix project name | `PHOENIX_PROJECT_NAME = "context-engineered-rag"` in `config.py` |
| Index auto-created by | `src/stores/vector_store.py → _get_index()` |
| Phoenix auto-started by | `src/tracing.py → init_tracing()` |
