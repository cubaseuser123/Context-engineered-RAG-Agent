"""runner - execute scenarios for one arm and write the eval substrate.

One ``RunRecord`` per agent invocation, appended to
``runs/<run_id>/records.jsonl``, plus a manifest that makes the run
reproducible. Resume is scenario-granular: a scenario with missing turns is
re-run from the start rather than resumed mid-conversation, because a
multi-turn answer depends on the answers before it.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from evals.dataset import Dataset, Scenario
from src import llm
from src.config import (
    CORPUS_DIR,
    DATASET_VERSION,
    EVAL_TEMPERATURE,
    GENERATOR_MODEL,
    JUDGE_MODEL,
    MEMORY_DB_PATH,
    RUNS_DIR,
)
from src.documents import corpus_doc_ids
from src.graph import compile_agent
from src.models.run_record import (
    CostInfo,
    MemoryInfo,
    OutputInfo,
    RetrievalInfo,
    RetrievedChunk,
    RoutingInfo,
    RunRecord,
)
from src.observability import turn_span
from src.refusal import is_refusal
from src.run_config import RunConfig
from src.stores import memory_store
from src.token_utils import count_tokens

logger = logging.getLogger(__name__)


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def new_run_id(config: RunConfig, *, prefix: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [p for p in (prefix, config.slug, stamp) if p]
    return "-".join(parts)


def run_paths(run_id: str) -> dict[str, Path]:
    directory = RUNS_DIR / run_id
    return {
        "dir": directory,
        "records": directory / "records.jsonl",
        "manifest": directory / "manifest.json",
        "scorecard": directory / "scorecard.json",
        "judgements": directory / "judgements.jsonl",
    }


def _read_records(path: Path) -> list[RunRecord]:
    if not path.exists():
        return []
    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(RunRecord.model_validate_json(line))
        except Exception:
            logger.warning("Skipping unreadable record line: %s", line[:120])
    return records


def _write_records(path: Path, records: Iterable[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_jsonl() + "\n")


def _seed_memory(scenario: Scenario, user_id: str) -> None:
    memory_store.clear(user_id)
    for fact in scenario.setup.memory:
        memory_store.write(user_id, [fact], turn=-1)


def _build_record(
    *,
    run_id: str,
    config: RunConfig,
    scenario: Scenario,
    user_turn_index: int,
    question: str,
    state: dict,
    duration_ms: float,
    started_at: str,
    dataset: Dataset,
    sha: str | None,
) -> RunRecord:
    usage = state.get("synthesizer_usage") or {}
    router_usage = state.get("router_usage") or {}
    writer_usage = state.get("memory_writer_usage") or {}

    retrieved = state.get("retrieved_chunks")
    retrieval = None
    if retrieved is not None:
        retrieval = RetrievalInfo(
            query=state.get("retrieval_query"),
            k=state.get("retrieval_k"),
            filter=state.get("retrieval_filter_applied"),
            searches=int(state.get("searches", 0) or 0),
            chunk_ids=[c.get("chunk_id", "") for c in retrieved],
            chunks=[
                RetrievedChunk(
                    chunk_id=c.get("chunk_id", ""),
                    doc=c.get("source_doc", ""),
                    section=c.get("section", ""),
                    score=float(c.get("score", 0.0)),
                    vector_score=c.get("vector_score"),
                    evidence_scope=c.get("evidence_scope", "indexed_chunk"),
                    seed_chunk_ids=c.get("seed_chunk_ids", []),
                    expanded_chunk_ids=c.get("expanded_chunk_ids", []),
                    tokens=count_tokens(c.get("content", "")),
                )
                for c in retrieved
            ],
            retained_chunk_ids=[
                c.get("chunk_id", "") for c in (state.get("trimmed_chunks") or [])
            ],
        )

    routing = None
    if state.get("intent"):
        routing = RoutingInfo(
            intent=state.get("intent"),
            reasoning=state.get("router_reasoning"),
            department_requested=state.get("department_filter_requested"),
            department_applied=state.get("retrieval_filter_applied"),
            needs_retrieval=state.get("needs_retrieval"),
            search_query=state.get("search_query"),
            user_facts=state.get("user_facts", []),
        )

    response = state.get("response", "")
    timing = {
        "router": float(state.get("router_latency_ms") or 0.0),
        "retrieval": float(state.get("retrieval_latency_ms") or 0.0),
        "synthesizer": float(state.get("synthesizer_latency_ms") or 0.0),
        "memory_writer": float(state.get("memory_writer_latency_ms") or 0.0),
        "total": duration_ms,
    }

    return RunRecord(
        run_id=run_id,
        record_id=f"{scenario.id}:{user_turn_index:02d}",
        scenario_id=scenario.id,
        turn_index=user_turn_index,
        arm=config.arm,
        config=config.as_dict(),
        config_hash=config.config_hash,
        git_sha=sha,
        dataset_version=dataset.version,
        question=question,
        started_at=started_at,
        duration_ms=duration_ms,
        routing=routing,
        retrieval=retrieval,
        memory=MemoryInfo(
            read=[str(m.get("fact", "")) for m in (state.get("memory_entries") or [])],
            written=[
                str(m.get("fact", "")) for m in (state.get("new_memory_entries") or [])
            ],
            write_error=state.get("memory_write_error"),
        ),
        budget=state.get("budget_log"),
        prompt=state.get("prompt_text", "") or "",
        output=OutputInfo(
            response=response,
            citations=list(state.get("cited_sources") or []),
            unknown_citations=list(state.get("cited_unknown") or []),
            refused=is_refusal(response),
        ),
        cost=CostInfo(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            aux_prompt_tokens=int(router_usage.get("prompt_tokens", 0))
            + int(writer_usage.get("prompt_tokens", 0)),
            aux_completion_tokens=int(router_usage.get("completion_tokens", 0))
            + int(writer_usage.get("completion_tokens", 0)),
            model=GENERATOR_MODEL,
        ),
        timing=timing,
    )


def _error_record(
    *,
    run_id: str,
    config: RunConfig,
    scenario: Scenario,
    user_turn_index: int,
    question: str,
    started_at: str,
    duration_ms: float,
    dataset: Dataset,
    sha: str | None,
    error_type: str,
    message: str,
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        record_id=f"{scenario.id}:{user_turn_index:02d}",
        scenario_id=scenario.id,
        turn_index=user_turn_index,
        arm=config.arm,
        config=config.as_dict(),
        config_hash=config.config_hash,
        git_sha=sha,
        dataset_version=dataset.version,
        question=question,
        started_at=started_at,
        duration_ms=duration_ms,
        cost=CostInfo(model=GENERATOR_MODEL),
        error={"type": error_type, "message": message},
    )


def select_scenarios(
    dataset: Dataset,
    *,
    cohorts: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[Scenario]:
    scenarios = list(dataset.scenarios)
    if cohorts:
        scenarios = [s for s in scenarios if s.cohort in set(cohorts)]
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [s for s in scenarios if s.id in wanted]
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def execute_run(
    config: RunConfig,
    dataset: Dataset,
    *,
    run_id: str | None = None,
    cohorts: list[str] | None = None,
    scenario_ids: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    notes: str = "",
    invoke_turn=None,
) -> dict:
    """Run one arm over the selected scenarios. Returns paths and records."""
    from src.errors import classify  # local import keeps module import cheap

    run_id = run_id or new_run_id(config)
    paths = run_paths(run_id)
    scenarios = select_scenarios(
        dataset, cohorts=cohorts, scenario_ids=scenario_ids, limit=limit
    )
    if not scenarios:
        raise ValueError("No scenarios selected")

    sha = git_sha()
    app = compile_agent(config)
    corpus = set(corpus_doc_ids(CORPUS_DIR))

    existing: list[RunRecord] = _read_records(paths["records"]) if resume else []
    complete_ids = {
        scenario.id
        for scenario in scenarios
        if len([r for r in existing if r.scenario_id == scenario.id])
        == scenario.expected_turns
        and all(
            r.error is None for r in existing if r.scenario_id == scenario.id
        )
    }
    records = [r for r in existing if r.scenario_id in complete_ids]
    _write_records(paths["records"], records)

    paths["dir"].mkdir(parents=True, exist_ok=True)
    errors = 0
    started_run = datetime.now(timezone.utc).isoformat()

    for scenario in scenarios:
        if scenario.id in complete_ids:
            logger.info("Skipping completed scenario %s", scenario.id)
            continue

        user_id = f"eval__{scenario.id}__{run_id[-6:]}"
        _seed_memory(scenario, user_id)
        history = [
            {"role": turn.role, "content": turn.content}
            for turn in scenario.setup.history
        ]

        user_turn_index = 0
        for turn in scenario.turns:
            if turn.role == "assistant":
                history.append({"role": "assistant", "content": turn.content})
                continue

            question = turn.content
            started_at = datetime.now(timezone.utc).isoformat()
            state = {
                "query": question,
                "user_id": user_id,
                "conversation_history": list(history),
                "turn_number": user_turn_index,
                "run_config": config.as_dict(),
            }
            started = time.perf_counter()
            try:
                with (nullcontext() if invoke_turn else turn_span(
                    **{
                        "run_id": run_id,
                        "scenario_id": scenario.id,
                        "turn_index": user_turn_index,
                        "arm": config.arm,
                        "config_hash": config.config_hash,
                        "git_sha": sha or "unknown",
                    }
                )):
                    if invoke_turn:
                        result, trace_id = invoke_turn(app, state, scenario.id, user_turn_index)
                    else:
                        result, trace_id = app.invoke(state), None
            except Exception as exc:  # noqa: BLE001 - failures are data
                errors += 1
                error_type, message = classify(exc)
                logger.error("%s/%s turn %d failed: %s", scenario.id, config.arm,
                             user_turn_index, message)
                record = _error_record(
                    run_id=run_id,
                    config=config,
                    scenario=scenario,
                    user_turn_index=user_turn_index,
                    question=question,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    dataset=dataset,
                    sha=sha,
                    error_type=error_type,
                    message=message,
                )
                records.append(record)
                with paths["records"].open("a", encoding="utf-8") as handle:
                    handle.write(record.to_jsonl() + "\n")
                break

            duration_ms = (time.perf_counter() - started) * 1000.0
            record = _build_record(
                run_id=run_id,
                config=config,
                scenario=scenario,
                user_turn_index=user_turn_index,
                question=question,
                state=result,
                duration_ms=duration_ms,
                started_at=started_at,
                dataset=dataset,
                sha=sha,
            )
            records.append(record)
            record.trace_id = trace_id
            with paths["records"].open("a", encoding="utf-8") as handle:
                handle.write(record.to_jsonl() + "\n")

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": record.output.response})
            user_turn_index += 1

    memory_store.clear_eval_rows()

    manifest = {
        "run_id": run_id,
        "created_at": started_run,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "arm": config.arm,
        "config": config.as_dict(),
        "config_hash": config.config_hash,
        "git_sha": sha,
        "dataset_version": dataset.version,
        "dataset_hash": dataset.hash,
        "dataset_label": DATASET_VERSION,
        "scenarios": len(scenarios),
        "turns": len(records),
        "errors": errors,
        "models": {
            "generator": GENERATOR_MODEL,
            "judge": JUDGE_MODEL,
            "temperature": EVAL_TEMPERATURE,
        },
        "corpus_documents": len(corpus),
        "cache": llm.read_ledger().get(
            datetime.now(timezone.utc).date().isoformat(), {}
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "memory_db": str(MEMORY_DB_PATH),
        },
        "notes": notes,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Run %s complete: %d records, %d errors", run_id, len(records), errors)

    return {
        "run_id": run_id,
        "config": config,
        "dataset": dataset,
        "records": records,
        "manifest": manifest,
        "paths": paths,
    }
