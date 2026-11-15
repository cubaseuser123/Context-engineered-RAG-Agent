"""Ingest the corpus as a full replace and check the P1 acceptance criteria.

Done when: 40 policy + 12 distractor documents, chunk IDs returned by ``query()``,
and top-8 retrieval overruns the 3,000-token cap on the median lookup. The gate
runs over the 52 evaluation questions rather than five hand-picked probes,
because the study is the workload the budget has to fire on; the five probes are
still reported next to it.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys

from src.config import CORPUS_DIR, PROJECT_ROOT, RETRIEVAL_BUDGET, RETRIEVAL_TOP_K
from src.documents import corpus_stats
from src.stores.vector_store import ingest, namespace_stats, query
from src.token_utils import count_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ingest")

SAMPLE_QUERIES = [
    "How many days of annual leave can I carry forward into next year?",
    "What is the reimbursement limit for a domestic business trip?",
    "How quickly must a Sev-1 incident be escalated?",
    "What are the required approval steps for a new vendor contract?",
    "How long are interview records retained?",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-replace ingest of the corpus plus the P1 retrieval check."
    )
    parser.add_argument("--no-wipe", action="store_true", help="upsert without clearing the namespace first")
    return parser


def run_probes(queries: list[str], *, keep_ids: bool) -> list[dict]:
    results = []
    for text in queries:
        chunks = query(text, k=RETRIEVAL_TOP_K)
        entry = {
            "query": text,
            "chunks": len(chunks),
            "top8_tokens": count_chunks(chunks),
        }
        if keep_ids:
            entry["chunk_ids"] = [c["chunk_id"] for c in chunks]
        results.append(entry)
    return results


def eval_turn_questions() -> list[str]:
    """The questions the study actually asks, in dataset order."""
    try:
        from evals.dataset import load_dataset
        from src.config import SCENARIOS_PATH

        dataset = load_dataset(SCENARIOS_PATH)
    except Exception as exc:  # the corpus check should not depend on the dataset existing
        logger.warning("Evaluation dataset unavailable, gate falls back to sample queries: %s", exc)
        return []
    return [t.content for s in dataset.scenarios for t in s.turns if t.role == "user"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not CORPUS_DIR.exists():
        logger.error("Corpus directory missing: %s", CORPUS_DIR)
        return 1

    stats = corpus_stats(CORPUS_DIR)
    logger.info(
        "Corpus: %d docs, %d chunks, %d words, %d tokens",
        stats["documents"], stats["chunks"], stats["words"], stats["tokens"],
    )
    print(json.dumps({k: stats[k] for k in ("documents", "chunks", "words", "tokens", "by_department")}, indent=2))

    if stats["documents"] < 52:
        logger.warning(
            "Corpus has %d documents; the design calls for 40 policy + 12 distractors",
            stats["documents"],
        )

    ingestion = ingest(CORPUS_DIR, wipe=not args.no_wipe)

    probe = run_probes(SAMPLE_QUERIES, keep_ids=True)
    median_probe = statistics.median([p["top8_tokens"] for p in probe]) if probe else 0

    turns = eval_turn_questions()
    turn_probe = run_probes(turns, keep_ids=False) if turns else []
    median_turns = statistics.median([p["top8_tokens"] for p in turn_probe]) if turn_probe else 0

    if turn_probe:
        logger.info(
            "Eval turns: median top-8 %s tokens, %d of %d over the %d cap",
            median_turns,
            sum(p["top8_tokens"] > RETRIEVAL_BUDGET for p in turn_probe),
            len(turn_probe),
            RETRIEVAL_BUDGET,
        )
        gate_value, gate_label = median_turns, "median_eval_turn_top8_tokens"
    else:
        gate_value, gate_label = median_probe, "median_top8_tokens"

    report = {
        "corpus": {k: stats[k] for k in ("documents", "chunks", "words", "tokens", "by_department")},
        "ingestion": ingestion,
        "namespace": namespace_stats(),
        "retrieval_probe": probe,
        "median_top8_tokens": median_probe,
        "eval_turn_probe": turn_probe,
        "median_eval_turn_top8_tokens": median_turns,
        "eval_turns_over_budget": sum(p["top8_tokens"] > RETRIEVAL_BUDGET for p in turn_probe),
        "eval_turns": len(turn_probe),
        "retrieval_budget": RETRIEVAL_BUDGET,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "gate": gate_label,
        "overruns_retrieval_budget": gate_value > RETRIEVAL_BUDGET,
    }
    out = PROJECT_ROOT / "corpus" / "ingest_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)

    if not report["overruns_retrieval_budget"]:
        logger.warning(
            "%s is %s tokens against a %d cap: the budget will not fire",
            gate_label, gate_value, RETRIEVAL_BUDGET,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
