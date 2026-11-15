"""report - render reports/<run_id>.md.

Health comes first and is never omitted. Every number carries n and an interval.
The report names the cases where the budget hurt, because a report that hides
them is an anti-goal.
"""
from __future__ import annotations

from datetime import datetime, timezone

from evals.dataset import Dataset
from evals.metrics import METRIC_ORDER
from evals.pricing import LABEL as PRICE_LABEL
from evals.judges import KAPPA_GATE

COHORT_ORDER = ["single_doc", "trap", "multi_turn", "memory", "refusal", "overall"]

_LABELS = {
    "routing_accuracy": "routing accuracy",
    "retrieval_recall_at_k": "retrieval recall@k",
    "retrieval_precision_at_k": "retrieval precision@k",
    "citation_precision": "citation precision",
    "citation_recall": "citation recall",
    "budget_invariant": "budget invariant",
    "constraint_pass": "constraint pass",
    "answer_correctness": "answer correctness (judge)",
    "groundedness": "groundedness (judge)",
    "citation_support": "citation support (judge)",
    "tokens_per_answer": "synthesis tokens / turn",
    "latency_ms": "latency ms",
    "searches_per_answer": "searches / answer",
    "cost_usd": "synthesis cost USD (estimate)",
}


def _cell(metric: str, entry: dict | None) -> str:
    if not entry or not entry.get("n"):
        return "—"
    mean = entry["mean"]
    low, high = entry["ci_low"], entry["ci_high"]
    if metric in {"tokens_per_answer", "latency_ms", "searches_per_answer"}:
        return f"{mean:,.0f} [{low:,.0f}, {high:,.0f}] (n={entry['n']})"
    if metric == "cost_usd":
        return f"${mean:.5f} (n={entry['n']})"
    return f"{mean:.1%} [{low:.1%}, {high:.1%}] (n={entry['n']})"


def _health_table(scorecards: dict[str, dict]) -> str:
    rows = [
        ("scenario completion", "scenario_completion_rate", True),
        ("turn completion", "turn_completion_rate", True),
        ("failed turn rate", "api_error_rate", True),
        ("judge coverage", "judge_coverage", True),
        ("cache hit rate", "cache_hit_rate", True),
    ]
    lines = ["| Health | " + " | ".join(scorecards) + " |", "|---|" + "---|" * len(scorecards)]
    for label, key, as_pct in rows:
        cells = []
        for arm in scorecards:
            value = scorecards[arm]["health"].get(key, 0.0)
            cells.append(f"{value:.1%}" if as_pct else str(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| turns recorded | "
        + " | ".join(str(scorecards[arm]["health"].get("turns_recorded", 0)) for arm in scorecards)
        + " |"
    )
    return "\n".join(lines)


def _headline_table(scorecards: dict[str, dict], comparison: dict | None) -> str:
    arms = list(scorecards)
    lines = [
        "| Metric | " + " | ".join(arms) + " | delta (eng − naive) |",
        "|---|---|" + "---|" * max(0, len(arms) - 1) + "---|",
    ]
    for metric in METRIC_ORDER:
        cells = [
            _cell(metric, scorecards[arm]["cohorts"]["overall"]["metrics"].get(metric))
            for arm in arms
        ]
        delta_cell = "—"
        if comparison and comparison.get("available"):
            entry = comparison["metrics"].get(metric)
            if entry:
                delta_cell = (
                    f"{entry['delta']:+.3f} "
                    f"[{entry['ci_low']:+.3f}, {entry['ci_high']:+.3f}] "
                    f"(n={entry['n']})"
                )
                if "mcnemar" in entry:
                    delta_cell += f", McNemar p={entry['mcnemar']['p_value']:.3f}"
        lines.append(f"| {_LABELS.get(metric, metric)} | " + " | ".join(cells) + f" | {delta_cell} |")
    return "\n".join(lines)


def _cohort_tables(scorecards: dict[str, dict]) -> str:
    chunks = []
    for arm, scorecard in scorecards.items():
        lines = [f"### {arm}", "", "| Cohort | n | Metrics |", "|---|---|---|"]
        for cohort in COHORT_ORDER:
            entry = scorecard["cohorts"].get(cohort)
            if not entry:
                continue
            metrics = entry["metrics"]
            highlights = []
            for metric in ("answer_correctness", "constraint_pass", "retrieval_recall_at_k",
                           "citation_precision", "tokens_per_answer"):
                if metric in metrics:
                    highlights.append(
                        f"{_LABELS[metric]}: {_cell(metric, metrics[metric])}"
                    )
            lines.append(
                f"| {cohort} | {entry['n_turns']} | " + "; ".join(highlights) + " |"
            )
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def render_report(
    *,
    run_id: str,
    scorecards: dict[str, dict],
    dataset: Dataset,
    comparison: dict | None = None,
    manifest: dict | None = None,
    kappa: dict | None = None,
    sweep: dict | None = None,
    stability: float | None = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    expected_judgments = dataset.turns_per_arm * len(scorecards) * 3
    completed_judgments = sum(
        card.get("cohorts", {}).get("overall", {}).get("metrics", {}).get(metric, {}).get("n", 0)
        for card in scorecards.values()
        for metric in ("answer_correctness", "groundedness", "citation_support")
    )
    judging_status = (
        f"> QUALITY SCORING INCOMPLETE: {completed_judgments}/{expected_judgments} usable judgments. "
        "Partial judge metrics are not a whole-benchmark quality comparison."
        if completed_judgments < expected_judgments else ""
    )
    lines = [
        f"# Context engineering, measured — run `{run_id}`",
        "",
        f"Dataset `{dataset.version}` ({dataset.hash}), "
        f"{len(dataset.scenarios)} scenarios, {dataset.turns_per_arm} turns per arm. "
        f"Generated {now}.",
        judging_status,
        "",
        "## Health",
        "",
        _health_table(scorecards),
        "",
        "Health is reported first because a correctness number over a broken run is noise.",
        "",
        "## Headline",
        "",
        _headline_table(scorecards, comparison),
        "",
        f"Cost is {PRICE_LABEL}. Intervals are Wilson 95% for proportions and percentile "
        "bootstrap 95% for means; deltas are paired bootstrap over turns, with McNemar "
        "for binary outcomes.",
        "",
        "## Cohorts",
        "",
        _cohort_tables(scorecards),
        "",
    ]

    if sweep:
        lines += ["## Budget sweep", "", "| Budget | Correctness | Tokens/answer | Latency ms |",
                  "|---|---|---|---|"]
        for point in sweep.get("points", []):
            lines.append(
                f"| {point['budget']} | {point['correctness']:.1%} | "
                f"{point['tokens']:,.0f} | {point['latency_ms']:,.0f} |"
            )
        if sweep.get("knee") is not None:
            lines.append("")
            lines.append(f"Knee marked at budget **{sweep['knee']}**.")
        lines.append("")

    for arm, scorecard in scorecards.items():
        confusion = scorecard["refusal_confusion"]
        lines += [
            f"### Refusal confusion — {arm}",
            "",
            "| | warrant refusal | answer |",
            "|---|---|---|",
            f"| refused | {confusion['refused_warranted']} | {confusion['refused_unwarranted']} |",
            f"| answered | {confusion['answered_warranted']} | {confusion['answered_unwarranted']} |",
            "",
        ]

    failure_lines = ["## Failure gallery", ""]
    shown = 0
    for arm, scorecard in scorecards.items():
        for failure in scorecard.get("failures", []):
            if shown >= 10:
                break
            shown += 1
            failure_lines += [
                f"**{failure['scenario_id']}** turn {failure['turn_index']} "
                f"({arm}, {failure['cohort']}) — {', '.join(failure['reasons'])}",
                "",
                f"- run_id: `{failure['run_id']}`",
                f"- Q: {failure['question']}",
                f"- A: {failure['response'][:300]}",
                f"- retrieved: {', '.join(failure['retrieved_docs']) or '(none)'}",
                "",
                "<details><summary>context sent to the synthesizer</summary>",
                "",
                "```",
                failure["context"],
                "```",
                "",
                "</details>",
                "",
            ]
    if not shown:
        failure_lines.append("No code-evaluator failures in this run.")
    lines += failure_lines

    lines += ["## Judge calibration", ""]
    if kappa and kappa.get("available"):
        lines.append(
            f"Cohen's κ = **{kappa['kappa']:.2f}** on {kappa['n']} human-labelled samples "
            f"(gate: κ ≥ {KAPPA_GATE})."
        )
        if kappa["kappa"] < KAPPA_GATE:
            lines.append(
                "**Below the gate — judge numbers in this report are provisional and "
                "must not be quoted as findings.**"
            )
    else:
        lines.append(
            "Not yet calibrated. Judge metrics are reported but provisional until "
            f"κ ≥ {KAPPA_GATE} is computed on 60 human labels."
        )
    lines.append("")

    lines += [
        "## Caveats",
        "",
        "- The naive arm has no refusal path by design; it always retrieves and always "
        "answers. That is stock RAG, not a handicap.",
        "- Both arms share the model, prompt, temperature, index and embedding model. "
        "Only routing, memory and budget differ.",
        "- Naive prompts are ~6K tokens and do not overflow Qwen's window; its cost is "
        "token spend and latency, not truncation.",
    ]
    if stability is not None:
        lines.append(
            f"- Repeat-stability on the lookup cohort: **{stability:.1%}**. "
            + (
                "Below 90%, every comparison above carries that caveat."
                if stability < 0.9
                else "Above 90%, so run-to-run variation is not the story."
            )
        )
    if manifest:
        lines.append(
            f"- Code version `{(manifest.get('git_sha') or 'unknown')[:12]}`, "
            f"dataset hash `{manifest.get('dataset_hash')}`, judge "
            f"`{manifest.get('models', {}).get('judge')}`."
        )
    lines.append("")
    return "\n".join(lines)
