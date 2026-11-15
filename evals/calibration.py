"""calibration - 60 blind human labels, then Cohen's kappa per judge.

Selection is stratified by (cohort x arm) and weighted toward traps and refusals,
because that is where a judge is most likely to disagree with a human. The sheet
never contains the judge's verdict: a label made after seeing it is not a label.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from evals.dataset import Dataset
from evals.judges import KAPPA_GATE, JudgeResult, read_judgements
from evals.stats import cohens_kappa
from src.models.run_record import RunRecord

WEIGHTS = {"trap": 2.0, "refusal": 1.5, "multi_turn": 1.2, "memory": 1.2, "single_doc": 1.0}

SHEET_COLUMNS = [
    "record_id",
    "scenario_id",
    "turn_index",
    "arm",
    "cohort",
    "question",
    "response",
    "reference_answer",
    "human_score",
    "human_notes",
]


def select_samples(
    records: list[RunRecord], dataset: Dataset, *, n: int = 60, seed: int = 11
) -> list[dict]:
    """Stratified, deterministic, and blind to the judge."""
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.error is not None:
            continue
        try:
            scenario = dataset.by_id(record.scenario_id)
        except KeyError:
            continue
        groups[(scenario.cohort, record.arm)].append(record)

    if not groups:
        return []

    total_weight = sum(
        WEIGHTS.get(cohort, 1.0) * len(items) for (cohort, _), items in groups.items()
    )
    rng = random.Random(seed)
    samples: list[dict] = []

    for (cohort, arm), items in sorted(groups.items()):
        weight = WEIGHTS.get(cohort, 1.0) * len(items)
        quota = max(1, round(n * weight / total_weight))
        quota = min(quota, len(items))
        pool = sorted(items, key=lambda r: r.record_id)
        rng.shuffle(pool)
        for record in pool[:quota]:
            scenario = dataset.by_id(record.scenario_id)
            expected = scenario.expected_for(record.turn_index)
            samples.append(
                {
                    "record_id": record.record_id,
                    "scenario_id": record.scenario_id,
                    "turn_index": record.turn_index,
                    "arm": record.arm,
                    "cohort": cohort,
                    "question": record.question,
                    "response": record.output.response,
                    "reference_answer": (expected.reference_answer if expected else ""),
                    "human_score": "",
                    "human_notes": "",
                }
            )

    return samples[:n]


def write_sheet(samples: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)
    return path


def _read_sheet(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_kappa(sheet_path: Path, judgements_path: Path) -> dict:
    """Kappa per judge metric against the human labels already in the sheet."""
    rows = [
        row
        for row in _read_sheet(sheet_path)
        if str(row.get("human_score", "")).strip() in {"0", "1"}
    ]
    if not rows:
        return {
            "available": False,
            "reason": "no human_score values in the sheet",
            "gate": KAPPA_GATE,
        }

    judgements: list[JudgeResult] = read_judgements(judgements_path)
    by_metric: dict[str, dict[str, int | None]] = defaultdict(dict)
    for result in judgements:
        by_metric[result.metric][result.record_id] = result.score

    per_metric: dict[str, dict] = {}
    for metric, scores in by_metric.items():
        human: list[int] = []
        judge: list[int] = []
        for row in rows:
            judged = scores.get(row["record_id"])
            if judged is None:
                continue
            human.append(int(row["human_score"]))
            judge.append(int(judged))
        if human:
            per_metric[metric] = cohens_kappa(human, judge)

    primary = per_metric.get("answer_correctness") or next(
        iter(per_metric.values()), {"kappa": 0.0, "n": 0}
    )
    return {
        "available": True,
        "kappa": primary["kappa"],
        "n": primary["n"],
        "gate": KAPPA_GATE,
        "passes_gate": primary["kappa"] >= KAPPA_GATE,
        "per_metric": per_metric,
        "what_happens_next": (
            "quote judge numbers"
            if primary["kappa"] >= KAPPA_GATE
            else "revise the judge prompts and recalibrate before quoting"
        ),
    }


def write_kappa(result: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path
