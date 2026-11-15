"""metrics - deterministic evaluators and cohort aggregation.

Every metric carries its denominator. Binary metrics get Wilson intervals,
continuous ones get bootstrap intervals, and nothing is reported as a bare
percentage.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from evals import pricing
from evals.dataset import Dataset, Expected, Scenario
from evals.stats import (
    bootstrap_mean_ci,
    mcnemar_exact,
    paired_bootstrap_delta,
    proportion_interval,
)
from src.models.run_record import RunRecord

BINARY_METRICS = {
    "routing_accuracy",
    "budget_invariant",
    "constraint_pass",
    "answer_correctness",
    "groundedness",
    "citation_support",
}

CONTINUOUS_METRICS = {
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "citation_precision",
    "citation_recall",
    "tokens_per_answer",
    "latency_ms",
    "searches_per_answer",
    "cost_usd",
}

METRIC_ORDER = [
    "routing_accuracy",
    "retrieval_recall_at_k",
    "retrieval_precision_at_k",
    "citation_precision",
    "citation_recall",
    "budget_invariant",
    "constraint_pass",
    "answer_correctness",
    "groundedness",
    "citation_support",
    "tokens_per_answer",
    "latency_ms",
    "searches_per_answer",
    "cost_usd",
]


@dataclass
class TurnEvaluation:
    scenario_id: str
    cohort: str
    arm: str
    turn_index: int
    metrics: dict[str, float | bool | None] = field(default_factory=dict)
    refused: bool = False
    warranted_refusal: bool = False

    def numeric(self, name: str) -> float | None:
        value = self.metrics.get(name)
        if value is None:
            return None
        return 1.0 if value is True else 0.0 if value is False else float(value)


def constraint_pass(response: str, expected: Expected) -> bool:
    """``must_include`` as case-insensitive substrings, ``must_not_include`` as regex."""
    text = response or ""
    lowered = text.lower()
    for needle in expected.must_include:
        if needle.lower() not in lowered:
            return False
    for pattern in expected.must_not_include:
        try:
            if re.search(pattern, text):
                return False
        except re.error:
            if pattern.lower() in lowered:
                return False
    return True


def check_budget_invariant(
    budget: dict, retrieved: list[dict], retained_ids: list[str]
) -> bool:
    """Every zone within cap, total within cap, drops are the lowest-scoring."""
    if not budget:
        return True
    for zone in budget.get("zones", []):
        cap = zone.get("cap")
        if cap is not None and zone.get("tokens_after", 0) > cap:
            return False
    total_cap = budget.get("total_cap")
    if total_cap is not None and budget.get("total_after", 0) > total_cap:
        return False

    if retained_ids:
        retained_scores = [
            float(c.get("score", 0.0))
            for c in retrieved
            if c.get("chunk_id") in set(retained_ids)
        ]
        floor = min(retained_scores) if retained_scores else None
        for zone in budget.get("zones", []):
            if zone.get("zone_name") != "retrieval":
                continue
            for dropped in zone.get("dropped", []):
                score = dropped.get("score")
                if score is None or floor is None:
                    continue
                if float(score) > floor + 1e-9:
                    return False
    return True


def evaluate_turn(
    record: RunRecord,
    scenario: Scenario,
    user_turn_index: int,
    judge_scores: dict[str, int] | None = None,
) -> TurnEvaluation:
    """Score one recorded turn against its ground truth."""
    expected = scenario.expected_for(user_turn_index)
    evaluation = TurnEvaluation(
        scenario_id=scenario.id,
        cohort=scenario.cohort,
        arm=record.arm,
        turn_index=user_turn_index,
        refused=record.output.refused,
        warranted_refusal=bool(expected.refusal_warranted) if expected else False,
    )
    if record.error is not None:
        return evaluation

    retrieved_docs = (
        [chunk.doc for chunk in record.retrieval.chunks] if record.retrieval else []
    )
    retrieved_floor = set(retrieved_docs)

    if record.retrieval is not None:
        retained = list(record.retrieval.retained_chunk_ids)
        evaluation.metrics["budget_invariant"] = check_budget_invariant(
            record.budget or {},
            [chunk.model_dump() for chunk in record.retrieval.chunks],
            retained,
        )

    evaluation.metrics["tokens_per_answer"] = float(record.cost.total_tokens)
    evaluation.metrics["latency_ms"] = float(record.duration_ms)
    evaluation.metrics["searches_per_answer"] = float(
        record.retrieval.searches if record.retrieval else 0
    )
    cost = pricing.cost_usd(
        record.cost.model, record.cost.prompt_tokens, record.cost.completion_tokens
    )
    if cost is not None:
        evaluation.metrics["cost_usd"] = cost

    if expected is None:
        return evaluation

    if expected.intent and record.routing is not None:
        evaluation.metrics["routing_accuracy"] = (
            record.routing.intent == expected.intent
        )

    expected_docs = set(expected.docs)
    if expected_docs and record.retrieval is not None:
        evaluation.metrics["retrieval_recall_at_k"] = len(
            expected_docs & retrieved_floor
        ) / len(expected_docs)
        evaluation.metrics["retrieval_precision_at_k"] = (
            sum(1 for doc in retrieved_docs if doc in expected_docs) / len(retrieved_docs)
            if retrieved_docs
            else 0.0
        )

    answers = bool(record.output.response) and not record.output.refused
    if expected_docs and answers:
        cited = set(record.output.citations)
        unknown = record.output.unknown_citations
        denominator = len(cited) + len(unknown)
        evaluation.metrics["citation_precision"] = (
            len(cited & retrieved_floor) / denominator if denominator else 0.0
        )
        evaluation.metrics["citation_recall"] = len(expected_docs & cited) / len(
            expected_docs
        )

    if expected.must_include or expected.must_not_include:
        evaluation.metrics["constraint_pass"] = constraint_pass(
            record.output.response, expected
        )

    for metric, score in (judge_scores or {}).items():
        evaluation.metrics[metric] = bool(score)

    return evaluation


def collect(
    records: list[RunRecord],
    dataset: Dataset,
    judge_scores: dict[str, dict[str, int]] | None = None,
) -> list[TurnEvaluation]:
    evaluations: list[TurnEvaluation] = []
    for record in records:
        try:
            scenario = dataset.by_id(record.scenario_id)
        except KeyError:
            continue
        evaluations.append(
            evaluate_turn(
                record, scenario, record.turn_index, (judge_scores or {}).get(record.record_id)
            )
        )
    return evaluations


def _aggregate_metric(name: str, values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    if name in BINARY_METRICS:
        successes = sum(1 for v in values if v >= 0.5)
        interval = proportion_interval(successes, len(values))
    else:
        interval = bootstrap_mean_ci(values)
    return interval.as_dict()


def aggregate(evaluations: list[TurnEvaluation]) -> dict:
    grouped: dict[str, list[TurnEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        grouped[evaluation.cohort].append(evaluation)
        grouped["overall"].append(evaluation)

    result: dict[str, dict] = {}
    for group, items in grouped.items():
        metrics: dict[str, dict] = {}
        for name in METRIC_ORDER:
            values = [
                value
                for value in (item.numeric(name) for item in items)
                if value is not None
            ]
            if values:
                metrics[name] = _aggregate_metric(name, values)
        result[group] = {
            "n_turns": len(items),
            "n_scenarios": len({item.scenario_id for item in items}),
            "metrics": metrics,
        }
    return result


def refusal_confusion(evaluations: list[TurnEvaluation]) -> dict:
    """Refusal is the positive class: refusing when warranted is a true positive."""
    tp = sum(1 for e in evaluations if e.warranted_refusal and e.refused)
    fn = sum(1 for e in evaluations if e.warranted_refusal and not e.refused)
    fp = sum(1 for e in evaluations if not e.warranted_refusal and e.refused)
    tn = sum(1 for e in evaluations if not e.warranted_refusal and not e.refused)
    return {
        "refused_warranted": tp,
        "answered_warranted": fn,
        "refused_unwarranted": fp,
        "answered_unwarranted": tn,
        "n": tp + fn + fp + tn,
        "refusal_recall": tp / (tp + fn) if (tp + fn) else None,
        "refusal_precision": tp / (tp + fp) if (tp + fp) else None,
    }


def health(
    records: list[RunRecord],
    dataset: Dataset,
    manifest: dict | None = None,
    judged_ids: set[str] | None = None,
) -> dict:
    expected_scenarios = len(dataset.scenarios)
    expected_turns = dataset.turns_per_arm
    recorded_scenarios = {record.scenario_id for record in records}
    errors = [record for record in records if record.error is not None]
    completed_scenarios = 0
    for scenario in dataset.scenarios:
        turns = [r for r in records if r.scenario_id == scenario.id]
        if len(turns) == scenario.expected_turns and all(
            r.error is None for r in turns
        ):
            completed_scenarios += 1

    judged = len(judged_ids or set())

    cache = {}
    if manifest:
        cache = manifest.get("cache", {})
    calls = sum(bucket.get("calls", 0) for bucket in cache.values())
    cached = sum(bucket.get("cached_calls", 0) for bucket in cache.values())

    return {
        "scenarios_expected": expected_scenarios,
        "scenarios_completed": completed_scenarios,
        "scenario_completion_rate": (
            completed_scenarios / expected_scenarios if expected_scenarios else 0.0
        ),
        "turns_expected": expected_turns,
        "turns_recorded": len(records),
        "turn_completion_rate": (
            (len(records) - len(errors)) / expected_turns if expected_turns else 0.0
        ),
        "api_error_rate": len(errors) / len(records) if records else 0.0,
        "api_errors": len(errors),
        "judge_coverage": judged / len(records) if records else 0.0,
        "cache_hit_rate": cached / calls if calls else 0.0,
        "scenarios_seen": len(recorded_scenarios),
    }


def build_scorecard(
    *,
    arm: str,
    config: dict,
    config_hash: str,
    dataset: Dataset,
    records: list[RunRecord],
    manifest: dict | None = None,
    judge_scores: dict[str, dict[str, int]] | None = None,
    failures_limit: int = 10,
) -> dict:
    evaluations = collect(records, dataset, judge_scores)
    judged_ids = set(judge_scores or {})
    cohorts = aggregate(evaluations)
    failures = []
    for record in records:
        try:
            scenario = dataset.by_id(record.scenario_id)
        except KeyError:
            continue
        evaluation = evaluate_turn(
            record, scenario, record.turn_index, (judge_scores or {}).get(record.record_id)
        )
        reasons = []
        if record.error is not None:
            reasons.append(f"error:{record.error.type}")
        if evaluation.metrics.get("constraint_pass") is False:
            reasons.append("constraint_fail")
        if evaluation.metrics.get("retrieval_recall_at_k") == 0.0:
            reasons.append("retrieval_miss")
        if evaluation.metrics.get("budget_invariant") is False:
            reasons.append("budget_invariant")
        if evaluation.warranted_refusal and not evaluation.refused:
            reasons.append("failed_to_refuse")
        if not evaluation.warranted_refusal and evaluation.refused:
            reasons.append("refused_unnecessarily")
        if reasons:
            failures.append(
                {
                    "run_id": record.run_id,
                    "scenario_id": record.scenario_id,
                    "turn_index": record.turn_index,
                    "cohort": scenario.cohort,
                    "reasons": reasons,
                    "question": record.question,
                    "response": record.output.response[:600],
                    "citations": record.output.citations,
                    "retrieved_docs": [c.doc for c in (record.retrieval.chunks if record.retrieval else [])],
                    "context": record.prompt[:1200],
                }
            )

    return {
        "schema_version": "1",
        "arm": arm,
        "config": config,
        "config_hash": config_hash,
        "dataset_version": dataset.version,
        "dataset_hash": dataset.hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health": health(records, dataset, manifest, judged_ids),
        "cohorts": cohorts,
        "cohort_counts": dataset.cohort_counts(),
        "refusal_confusion": refusal_confusion(evaluations),
        "failures": failures[:failures_limit],
        "failure_count": len(failures),
    }


def compare_arms(
    records_by_arm: dict[str, list[RunRecord]],
    dataset: Dataset,
    judge_scores_by_arm: dict[str, dict[str, dict[str, int]]] | None = None,
) -> dict:
    """Paired comparison: bootstrap deltas for numbers, McNemar for outcomes."""
    if not {"naive", "engineered"} <= set(records_by_arm):
        return {"available": False, "reason": "both arms need records"}

    evals_by_arm = {
        arm: collect(records, dataset, (judge_scores_by_arm or {}).get(arm))
        for arm, records in records_by_arm.items()
    }
    per_turn: dict[str, dict[tuple[str, int], TurnEvaluation]] = {}
    for arm, evaluations in evals_by_arm.items():
        per_turn[arm] = {(e.scenario_id, e.turn_index): e for e in evaluations}

    shared = set(per_turn["naive"]) & set(per_turn["engineered"])
    comparison: dict[str, dict] = {}
    for name in METRIC_ORDER:
        pairs = []
        for key in sorted(shared):
            a = per_turn["naive"][key].numeric(name)
            b = per_turn["engineered"][key].numeric(name)
            if a is None or b is None:
                continue
            pairs.append((a, b))
        if not pairs:
            continue
        entry = paired_bootstrap_delta(pairs)
        if name in BINARY_METRICS:
            only_engineered = sum(
                1 for naive_val, eng_val in pairs if eng_val >= 0.5 and naive_val < 0.5
            )
            only_naive = sum(
                1 for naive_val, eng_val in pairs if eng_val < 0.5 and naive_val >= 0.5
            )
            entry["mcnemar"] = mcnemar_exact(only_engineered, only_naive)
        comparison[name] = entry

    return {
        "available": True,
        "n_paired_turns": len(shared),
        "metrics": comparison,
        "note": "delta = engineered - naive; positive means engineered is higher",
    }
