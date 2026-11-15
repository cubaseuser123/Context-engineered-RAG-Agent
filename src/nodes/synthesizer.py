"""synthesizer - the main generation node.

This is the only expensive call in the graph and the only one whose tokens are
counted as ``tokens_per_answer``. Both arms share the prompt and the model; the
only difference is the context string handed to it.
"""
from __future__ import annotations

import logging
import time

from src import llm
from src.citations import extract_citations
from src.config import CORPUS_DIR, EVAL_TEMPERATURE, SYNTHESIZER_MODEL
from src.documents import corpus_doc_ids
from src.models.state import AgentState
from src.observability import span
from src.prompts import EMPTY_CONTEXT_RESPONSE, SYNTHESIZER_SYSTEM_PROMPT
from src.run_config import RunConfig

logger = logging.getLogger(__name__)


def _build_context(state: AgentState) -> str:
    """Assemble the four context zones into a single user message."""
    parts = []
    memory = state.get("trimmed_memory", [])
    if memory:
        mem_text = "\n".join(f"- {m.get('fact', '')}" for m in memory)
        parts.append(f"=== MEMORY ===\n{mem_text}")

    chunks = state.get("trimmed_chunks", [])
    if chunks:
        doc_text = ""
        for chunk in chunks:
            doc_text += (
                f"\n[Source: {chunk.get('source_doc', 'unknown')}]\n"
                f"{chunk.get('content', '')}\n"
            )
        parts.append(f"=== DOCUMENTS ==={doc_text}")

    history = state.get("trimmed_history", [])
    if history:
        hist_text = "\n".join(
            f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history
        )
        parts.append(f"=== CONVERSATION ===\n{hist_text}")

    parts.append(f"==== CURRENT ====\n{state.get('query', '')}")
    return "\n\n".join(parts)


def synthesizer_node(state: AgentState) -> dict:
    """Generate the answer and extract the documents it actually cites."""
    config = RunConfig.from_dict(state.get("run_config"))
    chunks = state.get("trimmed_chunks", [])
    intent = state.get("intent", "")

    if config.memory and intent == "memory_recall" and state.get("user_facts") and not state.get("needs_retrieval", True):
        return {"response": "Thanks, I'll remember that: " + "; ".join(state["user_facts"]),
                "cited_sources": [], "cited_unknown": [], "prompt_text": _build_context(state),
                "synthesizer_usage": {}, "synthesizer_latency_ms": 0.0}

    # Only the engineered arm has a refusal path; the naive arm must always
    # answer, because that is the stock-RAG behaviour under study.
    if config.routing and intent == "policy_lookup" and not chunks:
        logger.info("Synthesizer: empty context for policy lookup, refusing")
        return {
            "response": EMPTY_CONTEXT_RESPONSE,
            "cited_sources": [],
            "cited_unknown": [],
            "prompt_text": _build_context(state),
            "synthesizer_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "synthesizer_latency_ms": 0.0,
        }

    context = _build_context(state)
    started = time.perf_counter()
    with span(
        "node.synthesizer",
        **{
            "node.name": "synthesizer",
            "model": SYNTHESIZER_MODEL,
            "arm": config.arm,
            "context.chars": len(context),
        },
    ) as annotate:
        response = llm.chat(
            [
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            model=SYNTHESIZER_MODEL,
            temperature=EVAL_TEMPERATURE,
        )
        annotate(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=response.latency_ms,
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    answer = response.text.strip()
    candidates = list(corpus_doc_ids(CORPUS_DIR))
    for chunk in chunks:
        doc = chunk.get("source_doc")
        if doc and doc not in candidates:
            candidates.append(doc)

    extraction = extract_citations(answer, candidates)

    with span(
        "node.synthesizer.citations",
        **{
            "citations.extracted": extraction.cited,
            "citations.unknown": extraction.unknown,
            "citations.markers": extraction.markers,
        },
    ):
        pass

    logger.info(
        "Synthesizer: %d chars, cited=%s, unknown=%s",
        len(answer),
        extraction.cited,
        extraction.unknown,
    )

    return {
        "response": answer,
        "cited_sources": extraction.cited,
        "cited_unknown": extraction.unknown,
        "prompt_text": context,
        "synthesizer_usage": response.usage.as_dict(),
        "synthesizer_latency_ms": latency_ms,
    }
