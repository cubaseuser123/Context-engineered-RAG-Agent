"""Run the engineered arm at 800 / 1,500 / 3,000 / unbounded over single-turn scenarios."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from evals.dataset import load_dataset
from evals.sweep import run_sweep
from src.config import CORPUS_DIR, SCENARIOS_PATH
from src.documents import corpus_doc_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_sweep")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", action="append", type=int, help="repeatable; omit for all four")
    parser.add_argument("--unbounded", action="store_true", help="include the unbounded point")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    dataset = load_dataset(SCENARIOS_PATH)
    problems = dataset.validate_against(set(corpus_doc_ids(CORPUS_DIR)))
    if problems:
        for problem in problems:
            logger.error("dataset: %s", problem)
        return 2

    budgets: tuple[int | None, ...]
    if args.budget:
        budgets = tuple(args.budget) + ((None,) if args.unbounded else ())
    else:
        budgets = (800, 1500, 3000) + ((None,) if args.unbounded else (None,))

    sweep = run_sweep(dataset, budgets=budgets, base_run_id=args.run_id, limit=args.limit)
    print(json.dumps({k: sweep[k] for k in ("sweep_id", "points", "knee", "path")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
