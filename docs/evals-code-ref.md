# Evals Code Reference

> [!IMPORTANT]
> This is Phase 2. All modules from the **Agent Code Ref**, **Context Engineering Code Ref**, and **Observability Code Ref** must be built and working first. You should have already run the 30-question test suite at least once so Phoenix has traces to pull from.

> [!NOTE]
> **What are evals and why do we need them?** In Phase 1, `run_test_suite.py` scored the agent with hardcoded string matching — it checked if the router returned the right intent and if the response mentioned the right source document. That works for binary checks, but it can't judge *quality*. "Did the agent actually answer the question correctly?" requires an LLM reading the response and making a judgment call. That's what evals are — using a second LLM (the "judge") to grade the first LLM's output. Phoenix gives us the infrastructure to run these judge calls at scale, log the results as annotations on traces, and compare runs side-by-side as experiments.

---

## What This Adds

Phase 2 introduces five evaluators that run against traces collected in Phoenix:

| Evaluator | Type | What it grades | Judge LLM? |
|---|---|---|---|
| **Router Accuracy** | Code-based | Did the router pick the correct intent? | No |
| **Source Coverage** | Code-based | Did the response cite the expected source documents? | No |
| **Answer Correctness** | LLM-as-judge | Is the agent's answer factually correct given the expected answer? | Yes (Gemini Flash) |
| **Hallucination** | LLM-as-judge | Did the agent invent information not present in the retrieved context? | Yes (Gemini Flash) |
| **Faithfulness** | LLM-as-judge | Is the response grounded in the retrieved documents? | Yes (Gemini Flash) |

The first two are **code evaluators** — pure Python, no LLM call, deterministic. The last three are **LLM evaluators** — they send the agent's input/output to Gemini Flash with a grading prompt and get back a label + explanation.

> [!TIP]
> Code evaluators are free and instant. LLM evaluators cost tokens but catch things code can't — like whether an answer is *semantically* correct even if it uses different wording than the expected answer. A good eval suite uses both.

---

## How Phoenix Evals Work (the mental model)

Here's the flow, step by step:

```
1. Agent runs → traces land in Phoenix (Phase 1 already does this)
2. You export those traces as a pandas DataFrame
3. You reshape the DataFrame so each row has the columns your evaluators need
4. You define evaluators (code-based or LLM-based)
5. You run evaluators against the DataFrame → get scores + labels + explanations
6. You log those scores back to Phoenix as "annotations" on the original spans
7. Open Phoenix UI → every trace now shows eval scores alongside the agent's execution
```

Steps 2–6 are what this code reference implements.

For experiments (comparing two runs), the flow adds one more layer:

```
8. Upload your test questions as a Phoenix "dataset"
9. Define a "task" function that runs the agent on each question
10. Run the experiment: Phoenix calls your task on every question, then runs evaluators
11. Phoenix UI shows a comparison table: Experiment A vs Experiment B, per question
```

---

## Folder Structure (additions to existing project)

```
context-engineered-rag/
├── src/
│   └── evals/                       # NEW — all eval logic lives here
│       ├── __init__.py
│       ├── extract.py               # pull traces from Phoenix → DataFrame
│       ├── evaluators.py            # 5 evaluators (2 code + 3 LLM)
│       └── experiment.py            # Phoenix experiment runner
│
├── scripts/
│   ├── run_evals.py                 # NEW — run all evals against latest traces
│   └── run_experiment.py            # NEW — run a named experiment with evals
│
└── (everything else from Phase 1)
```

---

## Dependencies (additions to `pyproject.toml`)

### `pyproject.toml`

Add `phoenix-evals` and `nest-asyncio` to your existing dependencies:

```toml
[project]
name = "context-engineered-rag"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.4",
    "langchain-google-genai",
    "langchain-pinecone",
    "pinecone",
    "arize-phoenix",
    "phoenix-evals",
    "openinference-instrumentation-langchain",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "tiktoken",
    "pydantic>=2.0",
    "python-dotenv",
    "google-generativeai",
    "nest-asyncio",
    "pandas",
]
```

```powershell
pip install -e .
```

> [!NOTE]
> `phoenix-evals` is a separate package from `arize-phoenix`. Phoenix is the server + UI, `phoenix-evals` is the evaluation SDK (LLM, ClassificationEvaluator, async_evaluate_dataframe, etc.). They're installed independently. `nest-asyncio` is needed because Phoenix's eval functions are async, and running them from a regular Python script (not a notebook) requires patching the event loop.

---

## Environment

No new env vars needed. The evals use the same `GOOGLE_API_KEY` that the agent already uses. The judge LLM (Gemini Flash) is the same model as the router — cheap and fast.

---

## Build Order: extract → evaluators → experiment → run_evals → run_experiment

> [!IMPORTANT]
> Build files in the exact order listed. Each module only imports from modules above it.

---

### `src/evals/__init__.py`

```python
"""evals — Phoenix evaluation pipeline for the context-engineered RAG agent."""
```

---

### `src/evals/extract.py`

Pulls traces from the running Phoenix server and reshapes them into a DataFrame that evaluators can consume. This is the bridge between "traces in Phoenix" and "rows in a table that we can score."

**Why this is its own module**: Extracting and reshaping trace data is fiddly — Phoenix stores everything as nested span attributes, and evaluators need flat columns like `query`, `response`, `retrieved_chunks`. Keeping extraction separate means evaluators never touch Phoenix internals.

```python
"""
extract — Pull traces from Phoenix and reshape into eval-ready DataFrames.
Takes: Phoenix project name. Returns: pandas DataFrame with one row per agent invocation.
Each row has: query, response, intent, cited_sources, retrieved_chunks, budget_enforced.
"""
import logging
from typing import Any

import pandas as pd
from phoenix.client import Client
from phoenix.client.types.spans import SpanQuery

from src.config import PHOENIX_PROJECT_NAME

logger = logging.getLogger(__name__)


def get_trace_dataframe(project_name: str = PHOENIX_PROJECT_NAME) -> pd.DataFrame:
    """
    Export all top-level spans from Phoenix as a DataFrame.

    Phoenix stores every node execution as a "span." The top-level span
    represents the full graph invocation. Child spans are individual nodes
    (router, retriever, etc.). We pull the top-level spans because they
    contain the aggregated inputs/outputs of the entire agent run.

    Returns a DataFrame with columns:
    - context.span_id: unique span identifier (used to log annotations back)
    - context.trace_id: groups all spans in one agent run
    - attributes.input.value: the raw input to the graph (contains query, user_id, etc.)
    - attributes.output.value: the raw output (contains response, cited_sources, etc.)
    - status_code: OK or ERROR
    - latency_ms: end-to-end latency
    """
    client = Client()

    # Pull all root-level spans (these are the graph invocations)
    # Root spans have no parent — they represent full agent runs
    spans_df = client.spans.get_spans_dataframe(
        project_name=project_name,
    )

    if spans_df is None or spans_df.empty:
        logger.warning("No spans found in Phoenix project '%s'", project_name)
        return pd.DataFrame()

    logger.info("Exported %d spans from Phoenix project '%s'", len(spans_df), project_name)
    return spans_df


def reshape_for_evals(
    spans_df: pd.DataFrame,
    test_suite: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Reshape raw Phoenix spans into a flat DataFrame for evaluators.

    What this does:
    1. Extracts `query` and `response` from the nested span attributes
    2. Extracts `intent`, `cited_sources`, `retrieved_chunks` from output attributes
    3. Optionally joins with the test suite ground truth (expected_intent,
       expected_sources, expected_answer) so evaluators have both the agent's
       output AND the correct answer in the same row.

    The join is done by matching the query text. This works because the test suite
    has unique questions — no two questions are the same string.

    Parameters:
        spans_df: Raw DataFrame from get_trace_dataframe()
        test_suite: Optional list of test question dicts from test_suite.json.
                    If provided, ground truth columns are added.

    Returns:
        DataFrame with columns: span_id, query, response, intent, cited_sources,
        retrieved_chunks_text, budget_enforced, and optionally: expected_intent,
        expected_sources, expected_answer, correct_answer_is_idk
    """
    if spans_df.empty:
        return pd.DataFrame()

    rows = []
    for _, span in spans_df.iterrows():
        # Phoenix stores inputs/outputs as dicts in these attribute columns
        input_val = span.get("attributes.input.value")
        output_val = span.get("attributes.output.value")

        if not isinstance(input_val, dict) or not isinstance(output_val, dict):
            continue

        query = input_val.get("query", "")
        if not query:
            continue

        row = {
            "span_id": span.get("context.span_id", ""),
            "trace_id": span.get("context.trace_id", ""),
            "query": query,
            "response": output_val.get("response", ""),
            "intent": output_val.get("intent", ""),
            "cited_sources": output_val.get("cited_sources", []),
            "retrieved_chunks_text": _extract_chunk_text(output_val),
            "budget_enforced": output_val.get("budget_log", {}).get("enforced", False),
        }
        rows.append(row)

    eval_df = pd.DataFrame(rows)

    # Join with ground truth if test suite is provided
    if test_suite and not eval_df.empty:
        gt_df = pd.DataFrame(test_suite)
        gt_df = gt_df.rename(columns={"question": "query"})
        gt_cols = ["query", "expected_intent", "expected_sources", "expected_answer", "correct_answer_is_idk"]
        gt_df = gt_df[[c for c in gt_cols if c in gt_df.columns]]
        eval_df = eval_df.merge(gt_df, on="query", how="left")

    logger.info("Reshaped %d rows for evaluation", len(eval_df))
    return eval_df


def _extract_chunk_text(output: dict) -> str:
    """
    Concatenate all retrieved chunk contents into a single string.
    This becomes the 'context' that faithfulness evaluators check against.

    We use trimmed_chunks (post-budget-enforcement) rather than retrieved_chunks
    (pre-enforcement) because the synthesizer only saw the trimmed version.
    The faithfulness question is: "Is the response grounded in what the model
    actually received?" — not what was retrieved before trimming.
    """
    chunks = output.get("trimmed_chunks", output.get("retrieved_chunks", []))
    if not chunks:
        return ""
    return "\n\n".join(
        f"[Source: {c.get('source_doc', 'unknown')}]\n{c.get('content', '')}"
        for c in chunks
        if isinstance(c, dict)
    )
```

> [!NOTE]
> The `reshape_for_evals` function joins traces with ground truth by matching on the query string. This only works because our test suite has 30 unique questions. In a production eval pipeline with duplicate queries, you'd join on trace ID or timestamp instead.

> [!NOTE]
> We use `trimmed_chunks` (post-budget-enforcement) for the faithfulness context, not `retrieved_chunks`. This is intentional — the synthesizer only saw the trimmed context, so faithfulness should be judged against what the model *actually received*, not everything that was retrieved.

---

### `src/evals/evaluators.py`

Five evaluators: 2 code-based (free, instant, deterministic) and 3 LLM-based (Gemini Flash judge calls).

**How evaluators work in Phoenix**: An evaluator is a function that takes a row from your DataFrame (with columns like `query`, `response`, `expected_answer`) and returns a score. Code evaluators do this with Python logic. LLM evaluators do this by sending a prompt to a judge LLM and parsing the response. Phoenix's `ClassificationEvaluator` handles the LLM plumbing — you just write the grading prompt.

```python
"""
evaluators — Five evaluators for the context-engineered RAG agent.
Two code-based (router accuracy, source coverage).
Three LLM-based (answer correctness, hallucination, faithfulness).
All evaluators return scores that Phoenix can log as span annotations.
"""
import logging

from phoenix.evals import ClassificationEvaluator, LLM

from src.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge LLM setup
# ---------------------------------------------------------------------------
# We use Gemini Flash as the judge — same model as the router.
# It's cheap (~$0.05 per 1M input tokens) and fast.
# The judge never needs to be a frontier model — it just needs to follow
# a grading rubric consistently. Flash is more than capable for that.
#
# Phoenix's LLM() wrapper handles the Google GenAI client internally.
# Setting GOOGLE_API_KEY in the environment is sufficient.
# ---------------------------------------------------------------------------

_judge_llm: LLM | None = None


def _get_judge() -> LLM:
    """
    Singleton for the judge LLM.
    Phoenix's LLM() reads GOOGLE_API_KEY from the environment automatically
    when provider='google'. No need to pass it explicitly.
    """
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(provider="google", model="gemini-2.5-flash")
        logger.info("Judge LLM initialized: google/gemini-2.5-flash")
    return _judge_llm


# ============================================================================
# EVALUATOR 1: Router Accuracy (code-based)
# ============================================================================
#
# What it checks: Did the router classify the user's query into the correct
# intent? This is a simple string comparison — no LLM needed.
#
# Why it matters: If the router misclassifies, everything downstream breaks.
# A policy_lookup routed as out_of_scope means the user gets a refusal instead
# of an answer. A clarification routed as policy_lookup means the retriever
# runs unnecessarily and returns irrelevant chunks.
#
# Requires columns: intent (from agent), expected_intent (from ground truth)
# Returns: 1.0 if match, 0.0 if mismatch
# ============================================================================

def eval_router_accuracy(row: dict) -> dict:
    """
    Code evaluator: compare agent's intent to expected intent.

    Returns a dict with 'score', 'label', and 'explanation' —
    the standard format Phoenix expects for annotations.
    """
    actual = row.get("intent", "")
    expected = row.get("expected_intent", "")

    if not expected:
        # No ground truth available — can't score
        return {"score": None, "label": "no_ground_truth", "explanation": "No expected_intent in test suite"}

    correct = actual == expected
    return {
        "score": 1.0 if correct else 0.0,
        "label": "correct" if correct else "incorrect",
        "explanation": f"Expected '{expected}', got '{actual}'"
    }


# ============================================================================
# EVALUATOR 2: Source Coverage (code-based)
# ============================================================================
#
# What it checks: Did the agent cite at least one of the expected source
# documents in its response?
#
# Why "coverage" not "accuracy": The agent might cite additional relevant
# documents beyond the expected ones — that's fine, not a penalty.
# We check for overlap, not exact match.
#
# Requires columns: cited_sources (from agent), expected_sources (from ground truth)
# Returns: 1.0 if any expected source is cited, 0.0 if none
# ============================================================================

def eval_source_coverage(row: dict) -> dict:
    """
    Code evaluator: check if cited sources overlap with expected sources.

    Uses case-insensitive substring matching because the agent might cite
    "Leave Policy" while the ground truth says "Leave Policy" with different
    casing or extra whitespace.
    """
    cited = row.get("cited_sources", [])
    expected = row.get("expected_sources", [])

    if not expected:
        return {"score": None, "label": "no_ground_truth", "explanation": "No expected_sources in test suite"}

    if not cited:
        return {"score": 0.0, "label": "no_citations", "explanation": f"Agent cited nothing, expected {expected}"}

    # Normalize for comparison
    cited_lower = {s.lower().strip() for s in cited if isinstance(s, str)}
    expected_lower = {s.lower().strip() for s in expected if isinstance(s, str)}

    # Check if any expected source appears in (or is a substring of) any cited source
    found = set()
    for exp in expected_lower:
        for cit in cited_lower:
            if exp in cit or cit in exp:
                found.add(exp)

    if found:
        coverage = len(found) / len(expected_lower)
        return {
            "score": coverage,
            "label": "covered" if coverage == 1.0 else "partial",
            "explanation": f"Found {len(found)}/{len(expected_lower)} expected sources. Cited: {list(cited_lower)}"
        }

    return {
        "score": 0.0,
        "label": "missing",
        "explanation": f"Expected {list(expected_lower)}, cited {list(cited_lower)} — no overlap"
    }


# ============================================================================
# EVALUATOR 3: Answer Correctness (LLM-as-judge)
# ============================================================================
#
# What it checks: Is the agent's answer factually correct, given the expected
# answer from the test suite?
#
# Why an LLM judge instead of string matching: The agent might say "employees
# get twelve casual leave days per year" while the expected answer says
# "12 days of casual leave per calendar year." These are semantically identical
# but would fail an exact string match. The judge LLM understands equivalence.
#
# The prompt template uses {{}} mustache syntax — Phoenix replaces these with
# column values from the DataFrame at runtime.
#
# Requires columns: input (query), output (response), reference (expected_answer)
# Returns: 1.0 for correct, 0.0 for incorrect
# ============================================================================

ANSWER_CORRECTNESS_TEMPLATE = """You are a strict grader evaluating a policy assistant's response.

You will be given:
- The user's question
- The assistant's response
- The expected correct answer

Your job is to determine if the assistant's response is CORRECT — meaning it conveys the same factual information as the expected answer, even if the wording is different.

Rules:
- "correct" means the response contains the key facts from the expected answer
- Minor wording differences are OK (e.g., "12 days" vs "twelve days")
- The response may contain additional correct context — that's fine, not a penalty
- "incorrect" means the response is missing key facts, states wrong facts, or contradicts the expected answer
- If the expected answer says the correct response is "I don't know" and the assistant says it doesn't have the information, that is CORRECT

[BEGIN DATA]
************
[Question]: {input}
************
[Assistant Response]: {output}
************
[Expected Answer]: {reference}
************
[END DATA]

Is the assistant's response correct or incorrect based on the expected answer?"""


def get_answer_correctness_evaluator() -> ClassificationEvaluator:
    """
    Build the answer correctness evaluator.

    ClassificationEvaluator is Phoenix's built-in wrapper that:
    1. Takes a prompt template with placeholder variables
    2. Fills in the variables from DataFrame columns
    3. Sends the filled prompt to the judge LLM
    4. Parses the LLM's response into one of the defined choices
    5. Returns a score based on the choice mapping

    The 'choices' dict maps label strings to numeric scores.
    Phoenix will try to match the LLM's response to one of these labels.
    """
    return ClassificationEvaluator(
        name="answer_correctness",
        llm=_get_judge(),
        prompt_template=ANSWER_CORRECTNESS_TEMPLATE,
        choices={"correct": 1.0, "incorrect": 0.0},
    )


# ============================================================================
# EVALUATOR 4: Hallucination Detection (LLM-as-judge)
# ============================================================================
#
# What it checks: Did the agent make up information that isn't in the
# retrieved documents?
#
# This is different from "answer correctness." An answer can be correct but
# hallucinated (the agent guessed right without evidence), or incorrect but
# not hallucinated (the retrieved docs contained wrong info and the agent
# faithfully reported it).
#
# The key difference: hallucination checks against the RETRIEVED CONTEXT,
# while correctness checks against the EXPECTED ANSWER.
#
# Requires columns: input (query), output (response), reference (retrieved context)
# Returns: 1.0 for factual (no hallucination), 0.0 for hallucinated
# ============================================================================

HALLUCINATION_TEMPLATE = """You are evaluating whether an AI assistant's response contains hallucinated information.

A "hallucination" means the response states facts, details, or claims that are NOT present in the provided reference context. The reference context is the set of documents the assistant was given to answer the question.

You will be given:
- The user's question
- The assistant's response
- The reference context (retrieved documents the assistant had access to)

Rules:
- "factual" means every claim in the response can be traced back to the reference context
- "hallucinated" means the response contains at least one claim that cannot be found in the reference context
- If the response says "I don't have that information" and the reference context indeed doesn't contain the answer, that is "factual"
- Invented phone numbers, email addresses, URLs, or specific figures not in the context are hallucinations
- General knowledge statements (like "please contact HR") are acceptable if the context mentions HR

[BEGIN DATA]
************
[Question]: {input}
************
[Assistant Response]: {output}
************
[Reference Context]: {reference}
************
[END DATA]

Is the assistant's response factual or hallucinated based on the reference context?"""


def get_hallucination_evaluator() -> ClassificationEvaluator:
    """
    Build the hallucination detection evaluator.

    NOTE: The 'reference' column for this evaluator should be the
    retrieved_chunks_text (what the model actually received), NOT the
    expected_answer from the test suite. We're checking grounding,
    not correctness.
    """
    return ClassificationEvaluator(
        name="hallucination",
        llm=_get_judge(),
        prompt_template=HALLUCINATION_TEMPLATE,
        choices={"factual": 1.0, "hallucinated": 0.0},
    )


# ============================================================================
# EVALUATOR 5: Faithfulness (LLM-as-judge)
# ============================================================================
#
# What it checks: Is the response ONLY using information from the retrieved
# context, without adding external knowledge?
#
# Faithfulness vs Hallucination — what's the difference?
# - Hallucination = "did the agent make stuff up?" (binary: yes/no)
# - Faithfulness = "how well does the response stick to the source material?"
#   (more nuanced: fully faithful / partially faithful / unfaithful)
#
# In practice, for a policy assistant, these overlap heavily. But faithfulness
# is slightly stricter — even adding true but unsupported context (like general
# knowledge about HR practices) counts as unfaithful, whereas it might not
# count as a hallucination.
#
# Requires columns: input (query), output (response), reference (retrieved context)
# Returns: 1.0 for faithful, 0.5 for partially faithful, 0.0 for unfaithful
# ============================================================================

FAITHFULNESS_TEMPLATE = """You are evaluating whether an AI policy assistant's response is faithful to the source documents it was given.

"Faithful" means the response ONLY uses information present in the reference context. It does not add external knowledge, personal opinions, or information from sources not provided.

You will be given:
- The user's question
- The assistant's response
- The reference context (the policy documents the assistant was given)

Rate the faithfulness:
- "faithful": Every statement in the response is supported by the reference context
- "partially_faithful": Most of the response is supported, but some minor details come from outside the context
- "unfaithful": The response significantly departs from or goes beyond the reference context

[BEGIN DATA]
************
[Question]: {input}
************
[Assistant Response]: {output}
************
[Reference Context]: {reference}
************
[END DATA]

Is the assistant's response faithful, partially_faithful, or unfaithful?"""


def get_faithfulness_evaluator() -> ClassificationEvaluator:
    """
    Build the faithfulness evaluator.

    This uses a 3-way classification instead of binary. The 'partially_faithful'
    middle ground is important — it catches responses that are mostly correct
    but add a small detail from general knowledge. In a policy assistant,
    even small additions can be problematic (e.g., "HR is typically on the 3rd floor"
    when the policy doc doesn't mention floor numbers).
    """
    return ClassificationEvaluator(
        name="faithfulness",
        llm=_get_judge(),
        prompt_template=FAITHFULNESS_TEMPLATE,
        choices={"faithful": 1.0, "partially_faithful": 0.5, "unfaithful": 0.0},
    )


# ---------------------------------------------------------------------------
# Convenience: get all evaluators at once
# ---------------------------------------------------------------------------

def get_all_evaluators() -> dict:
    """
    Return all evaluators as a dict keyed by name.

    Code evaluators are plain functions.
    LLM evaluators are ClassificationEvaluator instances.

    The caller (run_evals.py) handles running them differently:
    - Code evaluators: loop over rows, call function directly
    - LLM evaluators: use Phoenix's async_evaluate_dataframe for concurrency
    """
    return {
        "router_accuracy": eval_router_accuracy,
        "source_coverage": eval_source_coverage,
        "answer_correctness": get_answer_correctness_evaluator(),
        "hallucination": get_hallucination_evaluator(),
        "faithfulness": get_faithfulness_evaluator(),
    }
```

> [!NOTE]
> **Why Gemini Flash as the judge?** Judge LLMs don't need to be the best model — they need to be *consistent*. Flash follows rubrics reliably and costs ~20x less than Pro. OpenAI's GPT-4o is the industry default for judges, but using Gemini keeps us on one provider and avoids a second API key. If you find Flash is too lenient or inconsistent, swap to `gemini-2.5-pro` by changing one string.

> [!NOTE]
> **Template variable naming**: Phoenix's `ClassificationEvaluator` uses `{input}`, `{output}`, and `{reference}` as the standard template variables. When you call `async_evaluate_dataframe`, you map your DataFrame columns to these variables. The mapping is: `input` ← query column, `output` ← response column, `reference` ← whatever ground truth or context the evaluator needs.

---

### `src/evals/experiment.py`

Phoenix experiment runner. Uploads the test suite as a dataset, defines a task function that runs the agent, and executes the experiment with all evaluators.

**What's an experiment?** An experiment is "run this function on every row of a dataset and evaluate the results." Phoenix stores the results and lets you compare multiple experiments side by side. This is how you A/B test prompt changes, model swaps, or parameter tuning — run experiment A with the old config, experiment B with the new config, compare scores.

```python
"""
experiment — Phoenix experiment runner.
Uploads test suite as a dataset, runs the agent as a task, evaluates with all 5 evaluators.
Results appear in the Phoenix UI under the Experiments tab.
"""
import json
import logging
from typing import Any

from phoenix.client import Client
from phoenix.client.experiments import create_evaluator, run_experiment

from src.config import TEST_SUITE_PATH, PHOENIX_PROJECT_NAME
from src.evals.evaluators import (
    eval_router_accuracy,
    eval_source_coverage,
    get_answer_correctness_evaluator,
    get_hallucination_evaluator,
    get_faithfulness_evaluator,
)
from src.graph import compile_agent, run_query
from src.tracing import init_tracing

logger = logging.getLogger(__name__)


def load_test_suite() -> list[dict]:
    """Load the 30-question test suite from disk."""
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_dataset(
    client: Client,
    test_suite: list[dict],
    dataset_name: str = "meridian-policy-evals",
) -> Any:
    """
    Upload the test suite as a Phoenix dataset.

    Phoenix datasets are tables where each row is one test case.
    The 'input' keys become what the task function receives.
    The 'output' keys become the expected output (ground truth).
    The 'metadata' keys are extra context for evaluators.

    We map:
    - input = {"query": question, "user_id": "eval_user"}
    - output = {"expected_intent": ..., "expected_sources": ..., "expected_answer": ...}
    - metadata = {"category": ..., "correct_answer_is_idk": ..., "question_id": ...}
    """
    import pandas as pd

    # Reshape test suite into Phoenix dataset format
    rows = []
    for q in test_suite:
        rows.append({
            "input.query": q["question"],
            "input.user_id": "eval_user",
            "output.expected_intent": q["expected_intent"],
            "output.expected_sources": json.dumps(q["expected_sources"]),
            "output.expected_answer": q["expected_answer"],
            "metadata.category": q["category"],
            "metadata.correct_answer_is_idk": q.get("correct_answer_is_idk", False),
            "metadata.question_id": q["id"],
        })

    df = pd.DataFrame(rows)

    dataset = client.datasets.create_dataset(
        dataframe=df,
        name=dataset_name,
        input_keys=["input.query", "input.user_id"],
        output_keys=["output.expected_intent", "output.expected_sources", "output.expected_answer"],
        metadata_keys=["metadata.category", "metadata.correct_answer_is_idk", "metadata.question_id"],
    )

    logger.info("Uploaded dataset '%s' with %d examples", dataset_name, len(rows))
    return dataset


def create_agent_task():
    """
    Create the task function that Phoenix calls for each test case.

    A task function receives a single 'example' dict (one row from the dataset)
    and returns the agent's output as a dict. Phoenix stores the output
    alongside the input for evaluators to score.

    The task function:
    1. Extracts the query from the example's input
    2. Runs the full agent graph
    3. Returns the fields that evaluators need
    """
    app = compile_agent()

    def agent_task(example: dict) -> dict:
        """Run the agent on one test question and return scorable output."""
        input_data = example.get("input", {})
        query = input_data.get("query", "")
        user_id = input_data.get("user_id", "eval_user")

        result = run_query(app, query=query, user_id=user_id, turn=0)

        # Build retrieved context string for faithfulness/hallucination evals
        chunks = result.get("trimmed_chunks", result.get("retrieved_chunks", []))
        context_text = "\n\n".join(
            f"[Source: {c.get('source_doc', 'unknown')}]\n{c.get('content', '')}"
            for c in chunks
            if isinstance(c, dict)
        )

        return {
            "response": result.get("response", ""),
            "intent": result.get("intent", ""),
            "cited_sources": result.get("cited_sources", []),
            "retrieved_context": context_text,
            "budget_enforced": result.get("budget_log", {}).get("enforced", False),
        }

    return agent_task


def build_experiment_evaluators() -> list:
    """
    Build evaluators in the format Phoenix experiments expect.

    Phoenix experiments use the @create_evaluator decorator to wrap
    evaluation functions. Each evaluator receives:
    - input: the test case input (query, user_id)
    - output: the task function's return value (response, intent, etc.)
    - expected: the ground truth from the dataset (expected_intent, etc.)

    Code evaluators return a score directly.
    LLM evaluators are wrapped to call the ClassificationEvaluator internally.
    """

    @create_evaluator(kind="code", name="router_accuracy")
    def router_eval(input: dict, output: dict, expected: dict) -> float:
        actual_intent = output.get("intent", "")
        expected_intent = expected.get("expected_intent", "")
        return 1.0 if actual_intent == expected_intent else 0.0

    @create_evaluator(kind="code", name="source_coverage")
    def source_eval(input: dict, output: dict, expected: dict) -> float:
        cited = output.get("cited_sources", [])
        expected_raw = expected.get("expected_sources", "[]")
        # expected_sources is JSON-encoded in the dataset
        if isinstance(expected_raw, str):
            expected_sources = json.loads(expected_raw)
        else:
            expected_sources = expected_raw

        if not expected_sources:
            return 1.0

        cited_lower = {s.lower().strip() for s in cited if isinstance(s, str)}
        expected_lower = {s.lower().strip() for s in expected_sources if isinstance(s, str)}

        for exp in expected_lower:
            for cit in cited_lower:
                if exp in cit or cit in exp:
                    return 1.0
        return 0.0

    @create_evaluator(kind="code", name="hallucination_check")
    def hallucination_code_eval(input: dict, output: dict, expected: dict) -> float:
        """
        Quick code-based hallucination check for IDK trap questions.
        The LLM evaluator handles the full hallucination check;
        this is a fast pre-filter for the obvious cases.
        """
        is_idk = expected.get("correct_answer_is_idk", False)
        if isinstance(is_idk, str):
            is_idk = is_idk.lower() == "true"
        if not is_idk:
            return 1.0  # Not a trap question, skip

        response = output.get("response", "").lower()
        idk_phrases = ["i don't have", "not in the available", "i can't find", "no information", "i don't know"]
        has_idk = any(phrase in response for phrase in idk_phrases)
        return 1.0 if has_idk else 0.0

    return [router_eval, source_eval, hallucination_code_eval]


def run_eval_experiment(
    experiment_name: str = "baseline",
    experiment_description: str = "Baseline agent evaluation",
    dataset_name: str = "meridian-policy-evals",
) -> Any:
    """
    Run a full evaluation experiment.

    This is the main entry point. It:
    1. Starts Phoenix + tracing
    2. Loads the test suite
    3. Uploads it as a Phoenix dataset (skips if already exists)
    4. Compiles the agent
    5. Runs the agent on every test question
    6. Scores each result with all evaluators
    7. Stores everything in Phoenix for UI viewing

    After this runs, open Phoenix (http://localhost:6006) → Experiments tab.
    You'll see a table with one row per question and columns for each evaluator's score.
    """
    # Initialize tracing so agent runs appear as traces
    init_tracing()

    client = Client()

    # Load and upload test suite
    test_suite = load_test_suite()
    dataset = upload_dataset(client, test_suite, dataset_name=dataset_name)

    # Build task and evaluators
    task = create_agent_task()
    evaluators = build_experiment_evaluators()

    # Run the experiment
    logger.info("Starting experiment '%s' with %d questions", experiment_name, len(test_suite))

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
    )

    logger.info("Experiment '%s' complete. View results at http://localhost:6006", experiment_name)
    return experiment
```

> [!IMPORTANT]
> The `create_evaluator` decorator comes from `phoenix.client.experiments`, NOT from `phoenix.evals`. These are two different eval systems in Phoenix: `phoenix.evals` works with DataFrames (batch-evaluate a table), while `phoenix.client.experiments` works with the experiment runner (evaluate as part of a structured experiment). We use the experiment system here because it gives us the comparison UI for free.

> [!TIP]
> To compare two configurations, run the experiment twice with different names:
> ```python
> run_eval_experiment(experiment_name="baseline", experiment_description="Default prompts")
> # ... change a prompt or model in config.py ...
> run_eval_experiment(experiment_name="v2-improved-router", experiment_description="Updated router prompt")
> ```
> Then open Phoenix → Experiments → select both → see a side-by-side comparison table.

---

### `scripts/run_evals.py`

Standalone script that runs all 5 evaluators against the latest traces in Phoenix and logs annotations back. This is the "quick eval" workflow — no experiment overhead, just score what's already there.

**When to use this vs run_experiment.py**: Use `run_evals.py` when you've already run the test suite (from Phase 1's `run_test_suite.py`) and just want to add eval scores to those existing traces. Use `run_experiment.py` when you want a fresh end-to-end run with built-in comparison.

```python
"""
run_evals — Run all evaluators against traces already in Phoenix.
Scores are logged back as annotations on the original spans.
Run: python -m scripts.run_evals
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

import nest_asyncio
import pandas as pd
from phoenix.client import Client
from phoenix.evals import async_evaluate_dataframe
from phoenix.evals.utils import to_annotation_dataframe

from src.config import PHOENIX_PROJECT_NAME, TEST_SUITE_PATH
from src.evals.extract import get_trace_dataframe, reshape_for_evals
from src.evals.evaluators import (
    eval_router_accuracy,
    eval_source_coverage,
    get_answer_correctness_evaluator,
    get_hallucination_evaluator,
    get_faithfulness_evaluator,
)
from src.tracing import init_tracing

# nest_asyncio patches the event loop so async functions work in regular scripts
# Without this, calling `await` from a non-async context would raise RuntimeError
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_suite() -> list[dict] | None:
    """Load test suite for ground truth joining. Returns None if file doesn't exist."""
    if not TEST_SUITE_PATH.exists():
        logger.warning("Test suite not found at %s — running without ground truth", TEST_SUITE_PATH)
        return None
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_code_evaluators(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the two code-based evaluators on every row.

    Code evaluators are plain Python functions — we just loop over rows.
    No async, no LLM calls, instant results.

    Returns the eval_df with new columns added:
    - router_accuracy_score, router_accuracy_label
    - source_coverage_score, source_coverage_label
    """
    router_scores = []
    source_scores = []

    for _, row in eval_df.iterrows():
        router_result = eval_router_accuracy(row.to_dict())
        source_result = eval_source_coverage(row.to_dict())
        router_scores.append(router_result)
        source_scores.append(source_result)

    eval_df["router_accuracy_score"] = [r["score"] for r in router_scores]
    eval_df["router_accuracy_label"] = [r["label"] for r in router_scores]
    eval_df["source_coverage_score"] = [r["score"] for r in source_scores]
    eval_df["source_coverage_label"] = [r["label"] for r in source_scores]

    # Print summary
    valid_router = [r["score"] for r in router_scores if r["score"] is not None]
    valid_source = [r["score"] for r in source_scores if r["score"] is not None]
    if valid_router:
        logger.info("Router accuracy: %.1f%% (%d/%d)", sum(valid_router) / len(valid_router) * 100, int(sum(valid_router)), len(valid_router))
    if valid_source:
        logger.info("Source coverage: %.1f%% (%d/%d)", sum(valid_source) / len(valid_source) * 100, int(sum(valid_source)), len(valid_source))

    return eval_df


async def run_llm_evaluators(eval_df: pd.DataFrame) -> None:
    """
    Run the three LLM-based evaluators using Phoenix's async batch evaluation.

    async_evaluate_dataframe sends all rows to the judge LLM concurrently
    (up to the concurrency limit). This is much faster than sequential calls —
    30 questions × 3 evaluators = 90 judge calls, which run in ~15 seconds
    with concurrency=10 vs ~3 minutes sequentially.

    After evaluation, results are converted to annotation format and logged
    back to Phoenix. Each annotation appears on the original span in the UI.
    """
    client = Client()

    # --- Prepare DataFrames for LLM evaluators ---
    # Phoenix's ClassificationEvaluator expects specific column names.
    # We need to create two separate DataFrames:
    # 1. For answer_correctness: input=query, output=response, reference=expected_answer
    # 2. For hallucination + faithfulness: input=query, output=response, reference=retrieved_context

    # DataFrame for answer correctness (reference = expected answer)
    correctness_df = eval_df[["span_id", "query", "response"]].copy()
    correctness_df = correctness_df.rename(columns={"span_id": "context.span_id"})

    if "expected_answer" in eval_df.columns:
        correctness_df["reference"] = eval_df["expected_answer"].fillna("")
        correctness_df["input"] = eval_df["query"]
        correctness_df["output"] = eval_df["response"]

        # Filter to rows that have ground truth
        correctness_df = correctness_df[correctness_df["reference"] != ""]

        if not correctness_df.empty:
            logger.info("Running answer correctness evaluator on %d rows...", len(correctness_df))
            correctness_evaluator = get_answer_correctness_evaluator()
            correctness_results = await async_evaluate_dataframe(
                dataframe=correctness_df,
                evaluators=[correctness_evaluator],
                concurrency=5,
            )

            # Log annotations back to Phoenix
            annotations = to_annotation_dataframe(correctness_results)
            client.spans.log_span_annotations_dataframe(dataframe=annotations)
            logger.info("Answer correctness annotations logged to Phoenix")
    else:
        logger.warning("No expected_answer column — skipping answer correctness eval")

    # DataFrame for hallucination + faithfulness (reference = retrieved context)
    grounding_df = eval_df[["span_id", "query", "response", "retrieved_chunks_text"]].copy()
    grounding_df = grounding_df.rename(columns={
        "span_id": "context.span_id",
        "retrieved_chunks_text": "reference",
    })
    grounding_df["input"] = eval_df["query"]
    grounding_df["output"] = eval_df["response"]

    # Filter to rows that have retrieved context
    grounding_df = grounding_df[grounding_df["reference"] != ""]

    if not grounding_df.empty:
        logger.info("Running hallucination + faithfulness evaluators on %d rows...", len(grounding_df))
        hallucination_evaluator = get_hallucination_evaluator()
        faithfulness_evaluator = get_faithfulness_evaluator()

        grounding_results = await async_evaluate_dataframe(
            dataframe=grounding_df,
            evaluators=[hallucination_evaluator, faithfulness_evaluator],
            concurrency=5,
        )

        # Log annotations back to Phoenix
        annotations = to_annotation_dataframe(grounding_results)
        client.spans.log_span_annotations_dataframe(dataframe=annotations)
        logger.info("Hallucination + faithfulness annotations logged to Phoenix")
    else:
        logger.warning("No retrieved context — skipping hallucination + faithfulness evals")


def print_summary(eval_df: pd.DataFrame) -> None:
    """Print a human-readable eval summary to stdout."""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total traces evaluated:   {len(eval_df)}")

    if "router_accuracy_score" in eval_df.columns:
        valid = eval_df["router_accuracy_score"].dropna()
        if len(valid) > 0:
            print(f"  Router accuracy:          {valid.mean() * 100:.1f}%")

    if "source_coverage_score" in eval_df.columns:
        valid = eval_df["source_coverage_score"].dropna()
        if len(valid) > 0:
            print(f"  Source coverage:           {valid.mean() * 100:.1f}%")

    print("-" * 60)
    print("  LLM eval results logged as annotations in Phoenix.")
    print("  Open http://localhost:6006 → click any trace → see eval scores.")
    print("=" * 60 + "\n")


async def main():
    # Start Phoenix (safe to call if already running)
    init_tracing()

    # Step 1: Export traces from Phoenix
    logger.info("Exporting traces from Phoenix...")
    spans_df = get_trace_dataframe()
    if spans_df.empty:
        logger.error("No traces found. Run the test suite first: python -m scripts.run_test_suite")
        sys.exit(1)

    # Step 2: Load ground truth and reshape
    test_suite = load_test_suite()
    eval_df = reshape_for_evals(spans_df, test_suite=test_suite)
    if eval_df.empty:
        logger.error("No evaluable rows after reshaping. Check trace format.")
        sys.exit(1)

    logger.info("Prepared %d rows for evaluation", len(eval_df))

    # Step 3: Run code evaluators (instant)
    eval_df = run_code_evaluators(eval_df)

    # Step 4: Run LLM evaluators (async, concurrent)
    await run_llm_evaluators(eval_df)

    # Step 5: Print summary
    print_summary(eval_df)

    logger.info("All evaluations complete. View annotations at http://localhost:6006")


if __name__ == "__main__":
    asyncio.run(main())
```

> [!NOTE]
> **Why two separate DataFrames for LLM evaluators?** Answer correctness checks the response against the *expected answer* (ground truth). Hallucination and faithfulness check the response against the *retrieved context* (what the model received). These are different reference texts, so they need different `reference` columns. Phoenix's `async_evaluate_dataframe` reads the `reference` column from whatever DataFrame you pass it.

> [!TIP]
> After running this script, open Phoenix → click any trace → look for the "Annotations" panel. You'll see labels like `answer_correctness: correct`, `hallucination: factual`, `faithfulness: faithful` alongside the normal trace data. This is the payoff — every trace now has eval scores attached.

---

### `scripts/run_experiment.py`

CLI entry point for running a named Phoenix experiment. This is the structured way to compare different agent configurations.

```python
"""
run_experiment — Run a named Phoenix experiment with the full eval suite.
Run: python -m scripts.run_experiment --name baseline --description "Default configuration"
Run: python -m scripts.run_experiment --name v2-router --description "Improved router prompt"
"""
import argparse
import logging

import nest_asyncio

from src.evals.experiment import run_eval_experiment

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run a Phoenix evaluation experiment against the test suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_experiment --name baseline
  python -m scripts.run_experiment --name v2-router --description "Updated router prompt with 4-shot examples"
  python -m scripts.run_experiment --name gemini-pro-judge --description "Using Gemini Pro as judge instead of Flash"

After running, open http://localhost:6006 → Experiments tab to see results.
To compare experiments, select multiple from the list.
        """,
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Experiment name (used in Phoenix UI). Use descriptive names like 'baseline', 'v2-router-prompt', 'pro-synthesizer'.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Evaluation experiment",
        help="Human-readable description of what this experiment tests.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="meridian-policy-evals",
        help="Phoenix dataset name. Reuse the same dataset across experiments for apples-to-apples comparison.",
    )

    args = parser.parse_args()

    logger.info("Running experiment: %s", args.name)
    experiment = run_eval_experiment(
        experiment_name=args.name,
        experiment_description=args.description,
        dataset_name=args.dataset_name,
    )
    logger.info("Done. Open http://localhost:6006 → Experiments to view results.")


if __name__ == "__main__":
    main()
```

---

## Running the Evals

### Option A — Score existing traces (quick, no re-running the agent)

```powershell
# 1. Make sure you've already run the test suite at least once (Phase 1)
python -m scripts.run_test_suite

# 2. Run evals against the traces that are now in Phoenix
python -m scripts.run_evals
```

This adds eval annotations to the traces already in Phoenix. Open `http://localhost:6006`, click any trace, and you'll see the eval scores.

### Option B — Run a structured experiment (re-runs the agent, stores results for comparison)

```powershell
# 1. Run the baseline experiment
python -m scripts.run_experiment --name baseline --description "Default prompts and config"

# 2. Make a change (e.g., edit a prompt in src/nodes/router.py)

# 3. Run a second experiment with the change
python -m scripts.run_experiment --name v2-improved-router --description "Added 4-shot examples to router prompt"

# 4. Open Phoenix → Experiments tab → select both → compare side by side
```

### What to look for in the Phoenix UI

| Section | What to check |
|---|---|
| **Traces → Annotations** | Each span now has eval labels (correct/incorrect, factual/hallucinated, faithful/unfaithful) |
| **Experiments → Table** | Per-question breakdown with scores for each evaluator |
| **Experiments → Compare** | Side-by-side diff showing which questions improved/regressed between experiments |
| **Experiments → Aggregate** | Overall percentages: router accuracy, source coverage, hallucination rate |

> [!TIP]
> **The debugging workflow**: Find a question where the agent scored `incorrect` on answer correctness. Click into that experiment row. See the agent's response, the expected answer, and the judge's explanation of why it scored incorrectly. Then click the corresponding trace to see the full node execution — which chunks were retrieved, what the budget enforcer trimmed, what the synthesizer prompt looked like. This trace-to-eval connection is why we built tracing in Phase 1.

---

## Changes from Original Spec

| Area | Spec (Phase 2 section of plan doc) | What We Changed | Why |
|---|---|---|---|
| **Judge LLM** | Not specified | Gemini Flash (`gemini-2.5-flash`) via `phoenix.evals.LLM(provider="google")` | Same provider as the agent, no second API key needed, cheap enough for 90+ judge calls |
| **Eval framework** | "LLM-as-a-judge" (generic) | Phoenix `ClassificationEvaluator` + `async_evaluate_dataframe` + experiments API | Phoenix's built-in eval SDK handles concurrency, annotation logging, and experiment comparison — no need to build this from scratch |
| **Hallucination eval** | "Tracked via whistleblower trap questions" | Two-layer: code-based IDK check (fast) + LLM hallucination judge (thorough) | Code check catches obvious traps instantly; LLM judge catches subtle hallucinations the code check misses |
| **Faithfulness eval** | Not explicitly mentioned | Added 3-way classification (faithful / partially_faithful / unfaithful) | Binary faithful/unfaithful is too coarse — "mostly right but added one unsupported detail" is a common failure mode worth distinguishing |
| **Eval runner** | "Same 30 questions with LLM-as-judge dropped in" | Two modes: `run_evals.py` (score existing traces) + `run_experiment.py` (fresh run with comparison) | Different workflows for different needs — quick check vs structured comparison |
| **Answer correctness scoring** | "Human-scored on first pass, LLM-as-judge in Phase 2" | `ClassificationEvaluator` with detailed rubric prompt replaces manual scoring | The whole point of Phase 2 — automate what was manual |
| **Experiment comparison** | "Named experiments in Phoenix" | Phoenix `run_experiment()` with `create_evaluator` decorators | Uses Phoenix's native experiment system rather than custom comparison logic |
