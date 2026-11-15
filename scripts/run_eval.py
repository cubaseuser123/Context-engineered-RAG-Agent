"""Run the two-arm evaluation and render a report.

    python -m scripts.run_eval --arm both --judge --report

Records land in ``runs/<run_id>/records.jsonl``. Judging is opt-in because it
costs a separate quota bucket; the report works without it and says so.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from evals.dataset import load_dataset
from evals.judges import judge_records, read_judgements, write_judgements
from evals.metrics import build_scorecard, compare_arms
from evals.report import render_report
from evals.runner import execute_run, new_run_id, run_paths
from src.config import CORPUS_DIR, REPORTS_DIR, SCENARIOS_PATH
from src.documents import corpus_doc_ids
from src.run_config import RunConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_eval")


def _parse_budget(value: str) -> int | None:
    return None if value.lower() in {"none", "unbounded", "inf"} else int(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["naive", "engineered", "both"], default="both")
    parser.add_argument("--budget", default="5000", help="engineered total budget, or 'unbounded'")
    parser.add_argument("--cohort", action="append", help="limit to a cohort (repeatable)")
    parser.add_argument("--scenario", action="append", help="limit to scenario ids (repeatable)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--judge", action="store_true", help="run the judge suite after generating")
    parser.add_argument("--judge-metrics", default="answer_correctness,groundedness,citation_support")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    dataset = load_dataset(SCENARIOS_PATH)
    problems = dataset.validate_against(set(corpus_doc_ids(CORPUS_DIR)))
    if problems:
        for problem in problems:
            logger.error("dataset: %s", problem)
        return 2

    base_id = args.run_id or new_run_id(RunConfig.engineered())
    arms = ["naive", "engineered"] if args.arm == "both" else [args.arm]
    scorecards: dict[str, dict] = {}
    records_by_arm: dict[str, list] = {}
    manifests: dict[str, dict] = {}
    judge_scores_by_arm: dict[str, dict[str, dict[str, int]]] = {}

    for arm in arms:
        config = (
            RunConfig.naive()
            if arm == "naive"
            else RunConfig.engineered(total_budget=_parse_budget(args.budget))
        )
        run_id = f"{base_id}-{arm}" if args.arm == "both" else base_id
        result = execute_run(
            config,
            dataset,
            run_id=run_id,
            cohorts=args.cohort,
            scenario_ids=args.scenario,
            limit=args.limit,
            resume=not args.no_resume,
            notes=args.notes,
        )
        records = result["records"]
        records_by_arm[arm] = records
        manifests[arm] = result["manifest"]

        paths = result["paths"]
        if args.judge:
            metrics = tuple(m.strip() for m in args.judge_metrics.split(",") if m.strip())
            existing = read_judgements(paths["judgements"])
            already = {(j.record_id, j.metric) for j in existing}
            new_judgements = judge_records(
                records, dataset, metrics=metrics, only_unjudged=already
            )
            if new_judgements:
                write_judgements(paths["judgements"], existing + new_judgements)
            scores: dict[str, dict[str, int]] = {}
            for judgement in read_judgements(paths["judgements"]):
                if judgement.score is not None:
                    scores.setdefault(judgement.record_id, {})[judgement.metric] = (
                        judgement.score
                    )
            judge_scores_by_arm[arm] = scores

        scorecard = build_scorecard(
            arm=config.arm,
            config=config.as_dict(),
            config_hash=config.config_hash,
            dataset=dataset,
            records=records,
            manifest=result["manifest"],
            judge_scores=judge_scores_by_arm.get(arm),
        )
        paths["scorecard"].write_text(
            json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        scorecards[arm] = scorecard
        logger.info("Wrote %s", paths["scorecard"])

    comparison = (
        compare_arms(records_by_arm, dataset, judge_scores_by_arm)
        if len(records_by_arm) == 2
        else None
    )

    if args.report:
        kappa_path = REPORTS_DIR / "judge_kappa.json"
        kappa = json.loads(kappa_path.read_text(encoding="utf-8")) if kappa_path.exists() else None
        report = render_report(
            run_id=base_id,
            scorecards=scorecards,
            dataset=dataset,
            comparison=comparison,
            manifest=next(iter(manifests.values()), None),
            kappa=kappa,
        )
        out = REPORTS_DIR / f"{base_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("Wrote %s", out)

    print(json.dumps({
        "run_id": base_id,
        "arms": list(scorecards),
        "comparison_available": bool(comparison and comparison.get("available")),
        "scorecards": {arm: str(run_paths(f"{base_id}-{arm}" if args.arm == "both" else base_id)["scorecard"]) for arm in scorecards},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
