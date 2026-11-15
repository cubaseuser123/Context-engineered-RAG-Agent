"""sweep - engineered arm at four budgets over the single-turn scenarios.

Produces the correctness-vs-tokens curve and marks the knee. The sweep is nearly
free because the budget is already a RunConfig parameter.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from evals.dataset import SINGLE_TURN_COHORTS, Dataset
from evals.metrics import build_scorecard
from evals.runner import execute_run, new_run_id, run_paths
from src.config import RUNS_DIR
from src.run_config import SWEEP_BUDGETS, RunConfig

logger = logging.getLogger(__name__)


def _overall(scorecard: dict, metric: str) -> float | None:
    entry = scorecard["cohorts"]["overall"]["metrics"].get(metric)
    return entry["mean"] if entry else None


def run_sweep(
    dataset: Dataset,
    *,
    budgets: tuple[int | None, ...] = SWEEP_BUDGETS,
    base_run_id: str | None = None,
    limit: int | None = None,
) -> dict:
    base_run_id = base_run_id or new_run_id(RunConfig.engineered(), prefix="sweep")
    points = []

    for budget in budgets:
        config = RunConfig.engineered(total_budget=budget)
        run_id = f"{base_run_id}-b{budget if budget is not None else 'unbounded'}"
        logger.info("Sweep point budget=%s run=%s", budget, run_id)
        result = execute_run(
            config,
            dataset,
            run_id=run_id,
            cohorts=list(SINGLE_TURN_COHORTS),
            limit=limit,
            notes=f"budget sweep point {budget}",
        )
        scorecard = build_scorecard(
            arm=config.arm,
            config=config.as_dict(),
            config_hash=config.config_hash,
            dataset=dataset,
            records=result["records"],
            manifest=result["manifest"],
        )
        correctness = _overall(scorecard, "answer_correctness")
        if correctness is None:
            correctness = _overall(scorecard, "constraint_pass")
        points.append(
            {
                "budget": budget if budget is not None else "unbounded",
                "run_id": run_id,
                "correctness": correctness or 0.0,
                "tokens": _overall(scorecard, "tokens_per_answer") or 0.0,
                "latency_ms": _overall(scorecard, "latency_ms") or 0.0,
                "n": scorecard["cohorts"]["overall"]["n_turns"],
                "scorecard_path": str(run_paths(run_id)["scorecard"]),
            }
        )

    numeric = [p for p in points if isinstance(p["budget"], int)]
    knee = None
    if numeric:
        best = max(p["correctness"] for p in numeric)
        for point in sorted(numeric, key=lambda p: p["budget"]):
            if point["correctness"] >= best - 0.05:
                knee = point["budget"]
                break

    sweep = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_id": base_run_id,
        "dataset_version": dataset.version,
        "dataset_hash": dataset.hash,
        "cohorts": list(SINGLE_TURN_COHORTS),
        "points": points,
        "knee": knee,
        "note": "knee = smallest budget within 5 correctness points of the best point",
    }

    path = RUNS_DIR / base_run_id / "sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    sweep["path"] = str(path)
    return sweep
