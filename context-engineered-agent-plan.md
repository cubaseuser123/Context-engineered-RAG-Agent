# Context-Engineered RAG Agent with Observability
## Project Plan — Phase 1 of a 3-Phase Learning Arc

---

## What This Is

A production-grade policy assistant agent built with LangGraph, instrumented with Arize Phoenix for full observability, and validated against a 30-question stress-test suite. The project is designed as a living foundation — Phase 1 delivers a working, observable, context-engineered agent. Phase 2 adds evals. Phase 3 adds an agent harness and production monitoring. Every architectural decision in Phase 1 is made with that arc in mind.

The agent answers questions about a synthetic company's internal policy documents. It routes queries, retrieves relevant policy chunks, manages a conversation memory store, and enforces a strict context budget at every step. Phoenix traces every node in the graph so you can see exactly what the agent did, why, and how much context it used.

---

## The Three Pillars

### Pillar 1 — Context Engineering

The agent does not naively stuff everything into the prompt. Every token that enters the context window is accounted for and intentional. There are four enforced context zones:

- **System prompt zone** — fixed, ~500 tokens. Contains the agent's identity, tool rules, and behavioral constraints. Never grows.
- **Memory zone** — capped at 500 tokens. Holds user-specific facts retrieved from the SQLite memory store. If memory exceeds the cap, oldest entries are dropped.
- **Retrieval zone** — capped at 3000 tokens. Holds policy chunks returned by the retriever. If retrieval returns more than the cap allows, chunks are ranked by relevance score and the lowest-scoring ones are dropped until the budget is met.
- **Conversation history zone** — capped at 1000 tokens. Holds the recent exchange. Older turns are summarized and compressed when the cap is approached.

The agent knows its budget. If a tool call would push total context over 5000 tokens, the agent must compress or drop before proceeding. This enforcement happens as an explicit node in the LangGraph graph, not as a post-hoc filter.

### Pillar 2 — Observability

Every step the agent takes is a Phoenix trace. The Phoenix server runs locally. Traces are collected automatically via OpenTelemetry instrumentation on LangGraph. No manual logging — the instrumentation is structural.

Each trace captures:

- Which router decision was made and why
- Which documents were retrieved and their relevance scores
- What the memory store returned
- How many tokens were in each context zone at the time of the LLM call
- The full model response and latency
- Whether the context budget was enforced and what was dropped

The Phoenix UI shows the full trace tree for every query. You can click into any node and see its inputs, outputs, and token counts. This is what makes the context engineering visible — not just claimed.

When evals are added in Phase 2, Phoenix becomes the trace store that evaluation runs pull from. The traces collected in Phase 1 become the source material for building the eval dataset.

### Pillar 3 — Quantified Validation

The project ships with a 30-question test suite designed to stress every component. Running the suite against the agent produces a scorecard with three dimensions per question:

- **Answer correctness** — did the agent get it right?
- **Source accuracy** — did it cite the right document?
- **Router accuracy** — did it route to the correct intent?

Each dimension is scored independently. The final output is a 3×30 matrix plus aggregate scores per dimension and per question category. This is the proof that the context engineering works — not anecdote, not vibes, numbers.

The suite is structured so that in Phase 2, LLM-as-a-judge evaluators and code-based evaluators can be dropped in on top of the same 30 questions without changing the test data.

---

## The Knowledge Base

### Synthetic Company: Meridian Technologies

A mid-sized B2B SaaS company, ~800 employees, HQ in Pune with offices in Bangalore and Hyderabad. The company context is specific enough to make policy questions realistic and ambiguous enough to create genuine retrieval challenges.

### 20 Policy Documents

Distributed across five departments. Each document is 600–900 words. Each department has at least one document containing a deliberate edge condition.

**HR (6 documents)**
1. Leave Policy — annual, sick, casual, maternity/paternity, bereavement
2. Work From Home Policy — eligibility, approval chain, equipment allowance
3. Performance Review Policy — cycle, rating scale, PIP process
4. Code of Conduct — workplace behavior, disciplinary ladder, zero-tolerance clauses
5. Referral Bonus Policy — *Edge: bonus payout split across two fiscal quarters depending on joining date — a single question spans two rules*
6. Probation and Confirmation Policy — *Edge: different notice periods apply during vs. after probation, and the document contradicts itself on one clause*

**IT & Security (4 documents)**
7. Device and Asset Policy — BYOD rules, company-issued equipment, disposal
8. Data Classification Policy — public / internal / confidential / restricted tiers
9. Acceptable Use Policy — internet, software, personal use boundaries
10. Incident Response Policy — *Edge: the reporting chain for a security incident differs based on severity level, and severity is defined in a different document (Data Classification)*

**Finance (4 documents)**
11. Expense Reimbursement Policy — categories, limits, submission deadlines
12. Travel Policy — booking class, per diem, advance request process
13. Vendor Payment Policy — approval thresholds, PO requirements
14. Budget Approval Policy — *Edge: approval authority depends on both budget category and amount — answering correctly requires reading a table across two sections*

**Legal & Compliance (3 documents)**
15. NDA and Confidentiality Policy — what employees can and cannot share
16. Intellectual Property Policy — who owns work created on company time vs. personal time
17. Whistleblower Policy — *Edge: the policy references an external ethics hotline but the hotline number is deliberately omitted — a trap for hallucination*

**Operations (3 documents)**
18. Office Access and Visitor Policy — badge access, visitor registration
19. Business Continuity Policy — remote work activation, communication chain
20. Procurement Policy — *Edge: procurement rules changed recently; the document has an old version clause still present alongside the new one — the agent must handle conflicting information in a single document*

---

## The Agent Architecture

### LangGraph Graph Structure

The agent is a stateful graph. State flows through nodes in sequence, with conditional edges that branch based on router decisions. There is no "always call the retriever" assumption — the router decides.

**Node: router**
Receives the raw user query plus conversation history. Classifies the query into one of four intents and routes accordingly. This is a fast, cheap call — Gemini Flash with a tight classification prompt and a structured output. Intents:

- `policy_lookup` — the user wants information from a policy document
- `clarification` — the user is following up on something already in conversation history
- `memory_recall` — the user is asking about something they previously told the agent
- `out_of_scope` — the query has nothing to do with company policy

**Node: retriever**
Only activated for `policy_lookup`. Takes the user query, runs a semantic search against the vector store of policy chunks, returns the top K chunks with relevance scores. Does not call the LLM.

**Node: context_budget_enforcer**
Runs after retrieval and memory fetch, before the synthesis call. Counts tokens across all four zones. If over budget, it trims: first drops low-scoring retrieval chunks, then compresses conversation history. Logs exactly what was dropped and why. This node never calls an LLM — it is pure logic.

**Node: memory_reader**
Queries the SQLite memory store for facts about the current user. Returns relevant entries up to the memory zone cap. Runs in parallel with the retriever for `policy_lookup` queries.

**Node: synthesizer**
The main generation call — Gemini Pro. Receives the assembled, budget-enforced context and produces the final answer with source citations. This is the only expensive LLM call in the graph.

**Node: memory_writer**
Runs after synthesis. Decides whether anything from this exchange is worth storing — user preferences, stated role, open questions. Writes to SQLite if yes. This call uses Gemini Flash with a short extraction prompt.

**Node: out_of_scope_handler**
Activated when the router returns `out_of_scope`. Returns a fixed polite refusal without calling the synthesizer. No tokens wasted.

### Memory Store

SQLite. One table. Each row is a memory entry: user ID, fact text, timestamp, source turn. The memory_reader fetches entries by user ID sorted by recency. The memory_writer inserts new entries. No embeddings, no vector search — memory is small enough that a full table scan per user is fine. This keeps memory transparent and debuggable.

### Vector Store

ChromaDB running locally. Policy documents are chunked at 512 tokens with 50-token overlap. Each chunk stores its source document name and section heading as metadata. The retriever filters by metadata when the router can infer a department from the query (e.g., "what is the travel policy" filters to Finance before semantic search).

---

## The 30-Question Test Suite

### Distribution

| Category | Count | What It Tests |
|---|---|---|
| Straightforward single-doc lookups | 8 | Basic retrieval and synthesis |
| Multi-document queries | 5 | Cross-document retrieval and assembly |
| Edge condition questions | 7 | The deliberate traps in the corpus |
| Memory-dependent questions | 4 | Memory store read and application |
| Out-of-scope questions | 3 | Router rejection accuracy |
| Clarification follow-ups | 3 | Conversation history use |

### Edge Condition Coverage

Every edge condition document has at least one question targeting its trap:
- Referral bonus split — question requires combining two date-dependent rules
- Probation contradiction — question about notice period during probation
- Incident response cross-reference — question requires reading severity from Data Classification, then applying Incident Response
- Budget approval table — question requires reading a row and column together
- Whistleblower hotline — question designed to elicit hallucination of a phone number
- Procurement version conflict — question about which rule currently applies

### Ground Truth Format

Each question has a pre-written ground truth record:
- Correct answer text
- Source document(s) that contain the answer
- Correct router intent
- Whether the question is expected to trigger context budget enforcement
- A flag for questions where "I don't know" or "I can't find that" is the correct answer

### Scoring

After running all 30 questions, the suite produces:

- **Router accuracy** — % of questions where the correct intent was selected
- **Source accuracy** — % of questions where the correct document was cited
- **Answer correctness** — human-scored on first run (binary correct/incorrect per question), later replaced by LLM-as-a-judge in Phase 2
- **Hallucination rate** — % of questions where the agent invented information not present in any document (primarily tracked via the whistleblower and other trap questions)
- **Context budget compliance** — % of calls where the enforcer had to intervene, and whether intervention produced a correct answer or degraded it

The project declares success when router accuracy ≥ 90%, source accuracy ≥ 85%, and hallucination rate = 0 on the trap questions.

---

## Tech Stack

| Component | Choice | Reason |
|---|---|---|
| Agent framework | LangGraph | Graph structure maps 1:1 to observable, evaluable components |
| LLM — synthesis | Gemini Pro via Vertex AI | Quality reasoning, GDG credits |
| LLM — router & memory writer | Gemini Flash via Vertex AI | Fast, cheap for classification and extraction |
| Vector store | ChromaDB (local) | Zero infrastructure, easy to inspect, swap-ready |
| Memory store | SQLite | Transparent, debuggable, no abstraction layer |
| Observability | Arize Phoenix (local server) | First-class LangGraph support, trace-to-eval pipeline for Phase 2 |
| Instrumentation | OpenTelemetry via `openinference-instrumentation-langchain` | Auto-instruments LangGraph nodes |
| Embeddings | `text-embedding-004` via Vertex AI | Consistent provider, no separate API key |
| Document format | Plain text .txt files | Simple to ingest, version-controllable |
| Test runner | Plain Python script | No framework overhead, easy to extend in Phase 2 |

---

## Phase 2 Readiness — What's Already Primed

Every decision above was made with the evals phase in mind. Specifically:

- **Phoenix traces from Phase 1 become the eval dataset.** The 30 test runs generate 30 traces with full node-level detail. Phase 2 turns these into labeled examples for router evals and skill evals without re-running the agent.
- **Router node is isolated.** Because routing is its own node with structured input/output, a code-based evaluator can assess every router decision independently — no need to re-run the full agent.
- **Synthesis node is isolated.** LLM-as-a-judge in Phase 2 evaluates the synthesizer's output given its inputs. The context budget enforcer ensures those inputs are always within a known bound, making judge prompts stable.
- **Ground truth records are eval-ready.** The source document field and expected answer field map directly to what Phoenix's eval framework expects for retrieval evals and answer correctness evals.
- **The 30-question suite runs as a structured experiment.** In Phase 2, the same suite is re-run with prompt changes or model swaps and results are compared as named experiments in Phoenix.

---

## Build Order

1. Generate the 20 synthetic policy documents and write the 30 test questions with ground truth records first — before any code. The corpus drives everything.
2. Set up the Vertex AI connection and verify Gemini Flash and Pro calls work.
3. Set up Phoenix server locally and verify traces appear.
4. Build the vector store — chunk documents, embed, load into ChromaDB.
5. Build the SQLite memory store with read and write operations.
6. Build the LangGraph graph — start with just router + synthesizer, no memory or budget enforcement yet. Verify traces in Phoenix.
7. Add the retriever node. Verify retrieval traces show chunk scores.
8. Add the context budget enforcer node. Verify enforcement logic triggers on long contexts.
9. Add memory reader and writer nodes.
10. Run the full 30-question suite. Score manually on first pass. Identify failure patterns in Phoenix traces.
11. Tune — fix prompt issues, chunking issues, or router prompt issues surfaced by the test run.
12. Re-run suite. Produce final scorecard. Declare Phase 1 complete.

---

*Phase 2: Router evals, skill evals, trajectory evals, LLM-as-a-judge, structured experiments — all built on top of this foundation without touching the agent architecture.*
