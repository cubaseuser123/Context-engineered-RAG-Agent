"""Judge calibration: select 60 blind samples, then compute Cohen's kappa.

    python -m scripts.calibrate_judge --select
    # label evals/datasets/calibration.v1.sheet.csv by hand
    python -m scripts.calibrate_judge --score

The sheet never shows the judge's verdict. Below kappa 0.6 the judge numbers are
provisional, and the report says so.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from evals.calibration import compute_kappa, select_samples, write_kappa, write_sheet
from evals.dataset import load_dataset
from evals.runner import run_paths
from src.config import DATASETS_DIR, REPORTS_DIR, RUNS_DIR, SCENARIOS_PATH
from src.models.run_record import RunRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("calibrate")


def _load_records(run_ids: list[str] | None) -> list[RunRecord]:
    paths = (
        [run_paths(run_id)["records"] for run_id in run_ids]
        if run_ids
        else sorted(RUNS_DIR.glob("*/records.jsonl"))
    )
    records: list[RunRecord] = []
    for path in paths:
        if not Path(path).exists():
            logger.warning("No records at %s", path)
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(RunRecord.model_validate_json(line))
    return records


def _judgement_path(run_ids: list[str] | None) -> Path:
    if run_ids:
        return run_paths(run_ids[0])["judgements"]
    candidates = sorted(RUNS_DIR.glob("*/judgements.jsonl"))
    if not candidates:
        return RUNS_DIR / "missing" / "judgements.jsonl"
    merged = RUNS_DIR / "judgements.all.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for path in candidates:
            handle.write(path.read_text(encoding="utf-8"))
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--runs", nargs="*", default=None, help="run ids; default all runs")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--sheet", default=str(DATASETS_DIR / "calibration.v1.sheet.csv"))
    args = parser.parse_args(argv)

    if not (args.select or args.score):
        parser.error("choose --select or --score")

    if args.select:
        dataset = load_dataset(SCENARIOS_PATH)
        records = _load_records(args.runs)
        samples = select_samples(records, dataset, n=args.n)
        path = write_sheet(samples, Path(args.sheet))
        logger.info("Wrote %d blind samples to %s", len(samples), path)
        print(json.dumps({"sheet": str(path), "samples": len(samples)}, indent=2))
        return 0

    sheet = Path(args.sheet)
    if not sheet.exists():
        logger.error("No sheet at %s; run --select first", sheet)
        return 1
    result = compute_kappa(sheet, _judgement_path(args.runs))
    out = write_kappa(result, REPORTS_DIR / "judge_kappa.json")
    print(json.dumps(result, indent=2))
    logger.info("Wrote %s", out)
    return 0 if result.get("available") else 1


if __name__ == "__main__":
    sys.exit(main())
