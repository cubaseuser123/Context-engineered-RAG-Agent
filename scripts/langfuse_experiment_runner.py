"""langfuse_experiment_runner.py - Run the evaluation and link directly as a Langfuse Experiment.

Runs both arms (naive vs engineered), records full execution traces, evaluates with LLM judges,
and links each turn to the Langfuse dataset item for side-by-side Experiment comparisons.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

import os
load_dotenv()
logger = logging.getLogger("langfuse_experiment")

from evals.dataset import load_dataset
from evals.judges import judge_records, read_judgements, append_judgement, judge_version
from evals.metrics import build_scorecard, compare_arms
from evals.report import render_report
from evals.runner import execute_run, new_run_id, select_scenarios
from src.config import CORPUS_DIR, REPORTS_DIR, SCENARIOS_PATH, RUNS_DIR
from src.documents import corpus_doc_ids
from src.run_config import RunConfig
from src.tracing import get_client
from scripts.live_langfuse import make_invoker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("langfuse_experiment")


def upload_dataset(client, ds):
    dataset_name = f"scenarios.{ds.version}"
    logger.info("Setting up Langfuse Dataset: %s", dataset_name)
    try:
        client.create_dataset(
            name=dataset_name,
            description="Meridian Technologies RAG Policy Benchmark (30 scenarios, 52 turns)",
            metadata={"version": ds.version, "scenarios_count": len(ds.scenarios)},
        )
    except Exception:
        pass

    for scenario in ds.scenarios:
        for idx, turn in enumerate(scenario.turns):
            item_id = f"{scenario.id}:{idx:02d}"
            expected = scenario.expected_for(idx)
            ref_ans = expected.reference_answer if expected else ""
            must_include = expected.must_include if expected else []
            docs = expected.docs if expected else []

            try:
                client.create_dataset_item(
                    dataset_name=dataset_name,
                    id=item_id,
                    input={"question": turn.content, "scenario_id": scenario.id, "turn_index": idx},
                    expected_output={
                        "reference_answer": ref_ans,
                        "must_include": must_include,
                        "expected_docs": docs,
                    },
                    metadata={
                        "cohort": scenario.cohort,
                        "tags": scenario.tags,
                        "scenario_id": scenario.id,
                        "turn_index": idx,
                    },
                )
            except Exception:
                pass
    client.flush()
    logger.info("Dataset %s ready in Langfuse.", dataset_name)
    return dataset_name


def main():
    os.environ["DISABLE_LLM_CACHE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["naive", "engineered", "both"], default="both")
    parser.add_argument("--budget", default="5000")
    parser.add_argument("--judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--run-id", help="Base run ID, for resuming a saved comparison")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", action="append", help="Run only these scenario IDs (repeatable)")
    args = parser.parse_args()

    client = get_client()
    if not client:
        logger.error("Langfuse client could not be initialized!")
        return 1

    dataset = load_dataset(SCENARIOS_PATH)
    problems = dataset.validate_against(set(corpus_doc_ids(CORPUS_DIR)))
    if problems:
        for p in problems:
            logger.error("Dataset validation error: %s", p)
        return 2

    # 1. Register Dataset
    dataset_name = upload_dataset(client, dataset)
    if args.scenario or args.limit is not None:
        if args.scenario:
            for scenario_id in args.scenario:
                dataset.by_id(scenario_id)  # fail on a typo rather than silently skip it
        dataset = dataset.model_copy(update={"scenarios": select_scenarios(dataset, scenario_ids=args.scenario, limit=args.limit)})

    # 2. Run Eval
    base_id = args.run_id or new_run_id(RunConfig.engineered(), prefix="smoke-v2" if args.scenario else "")
    arms = ["naive", "engineered"] if args.arm == "both" else [args.arm]
    records_by_arm = {}
    judge_scores_by_arm = {}
    scorecards = {}
    manifests = {}

    for arm in arms:
        config = RunConfig.naive() if arm == "naive" else RunConfig.engineered(total_budget=int(args.budget))
        run_id = f"{base_id}-{arm}"
        logger.info("=== Starting Execution: %s (arm=%s) ===", run_id, arm)

        result = execute_run(config, dataset, run_id=run_id, limit=args.limit, resume=args.resume,
                             scenario_ids=args.scenario,
                             invoke_turn=make_invoker(client, dataset_name, run_id, arm))
        records = result["records"]
        records_by_arm[arm] = records
        manifests[arm] = result["manifest"]
        paths = result["paths"]

        if args.judge:
            logger.info("Running LLM judges for %s...", arm)
            metrics = ("answer_correctness", "groundedness", "citation_support")
            existing = read_judgements(paths["judgements"])
            already = {(j.record_id, j.metric) for j in existing if j.score is not None and j.judge_version == judge_version(j.metric)}
            judge_records(records, dataset, metrics=metrics, only_unjudged=already,
                          on_result=lambda j: append_judgement(paths["judgements"], j))

            scores = {}
            for j in read_judgements(paths["judgements"]):
                if j.score is not None and j.judge_version == judge_version(j.metric):
                    scores.setdefault(j.record_id, {})[j.metric] = j.score
            judge_scores_by_arm[arm] = scores

        if not args.judge:
            scores = {}
            for j in read_judgements(paths["judgements"]):
                if j.score is not None and j.judge_version == judge_version(j.metric):
                    scores.setdefault(j.record_id, {})[j.metric] = j.score
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
        paths["scorecard"].write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
        scorecards[arm] = scorecard

        # Stream directly to Langfuse
        from scripts.score_langfuse_records import publish_resource_metrics
        judge_reasons = {(j.record_id, j.metric): j.reason for j in read_judgements(paths["judgements"])}
        score_ids = {}
        if args.resume and records:
            from scripts.langfuse_mcp import LangfuseMCP
            mcp = LangfuseMCP()
            cursor = None
            while True:
                params = {"limit": 100, "traceId": [r.trace_id for r in records],
                          "name": ["answer_correctness", "groundedness", "citation_support"]}
                if cursor:
                    params["cursor"] = cursor
                response = mcp.call("listScores", params)
                for score in response["data"]:
                    score_ids[(score["subject"]["id"], score["name"])] = score["id"]
                cursor = response.get("meta", {}).get("cursor")
                if not cursor:
                    break
        for rec in records:
            if not rec.trace_id:
                raise RuntimeError(f"Missing live trace for {rec.record_id}")
            for metric, value in judge_scores_by_arm.get(arm, {}).get(rec.record_id, {}).items():
                score_id = score_ids.get((rec.trace_id, metric)) or hashlib.sha256(f"{rec.trace_id}:{metric}".encode()).hexdigest()[:32]
                client.create_score(score_id=score_id, trace_id=rec.trace_id, name=metric, value=float(value),
                                    comment=judge_reasons.get((rec.record_id, metric), ""),
                                    environment="sdk-experiment")
            publish_resource_metrics(client, rec.model_dump())
        client.flush()

    comparison = compare_arms(records_by_arm, dataset, judge_scores_by_arm) if len(records_by_arm) == 2 else None
    if args.report:
        report = render_report(run_id=base_id, scorecards=scorecards, dataset=dataset, comparison=comparison, manifest=next(iter(manifests.values()), None))
        out = REPORTS_DIR / f"{base_id}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info("Wrote final report to %s", out)

    print("\n==========================================")
    print(f"EVALUATION COMPLETE: {base_id}")
    print("Everything tracked live inside Langfuse Cloud.")
    print("==========================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
