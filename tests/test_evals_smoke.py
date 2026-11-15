"""evals_smoke - cheap regression gate against the committed baseline.

Run with: pytest -m evals_smoke
Executes six scenarios, rebuilds the scorecard and fails if any cohort metric
falls below its baseline interval. Skips cleanly when keys or baseline are absent,
so the default test run stays offline and free.
"""
import json

import pytest

from evals.dataset import load_dataset
from evals.metrics import build_scorecard
from evals.runner import execute_run, run_paths
from src import llm
from src.config import BASELINE_PATH, SCENARIOS_PATH
from src.run_config import RunConfig


@pytest.mark.evals_smoke
def test_six_scenario_smoke_against_baseline():
    configured = llm.configured()
    if not configured["groq_keys"] or not configured["google_key"]:
        pytest.skip("provider keys not configured; the gate needs live models")
    if not BASELINE_PATH.exists():
        pytest.skip("no committed baseline yet: run a full eval, then copy its scorecard.json")

    dataset = load_dataset(SCENARIOS_PATH)
    subset = [scenario.id for scenario in dataset.scenarios[:6]]
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    config = RunConfig.engineered()

    result = execute_run(
        config,
        dataset,
        scenario_ids=subset,
        run_id="evals-smoke",
        resume=False,
        notes="pytest evals_smoke gate",
    )
    scorecard = build_scorecard(
        arm=config.arm,
        config=config.as_dict(),
        config_hash=config.config_hash,
        dataset=dataset,
        records=result["records"],
        manifest=result["manifest"],
    )

    failures = []
    for cohort, entry in scorecard["cohorts"].items():
        base = baseline.get("cohorts", {}).get(cohort)
        if not base:
            continue
        for metric, observed in entry["metrics"].items():
            reference = base.get("metrics", {}).get(metric)
            if not reference:
                continue
            if observed["mean"] < reference["ci_low"] - 0.001:
                failures.append(
                    cohort + "/" + metric + "=" + format(observed["mean"], ".3f")
                    + " below " + format(reference["ci_low"], ".3f")
                )
    if failures:
        pytest.fail(
            "regression vs baseline in " + result["run_id"]
            + " (" + str(run_paths(result["run_id"])["records"]) + "): "
            + ", ".join(failures)
        )
