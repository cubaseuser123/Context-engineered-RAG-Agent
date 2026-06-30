"""
run_experiment — Run a named Phoenix experiment with the full eval suite.
"""
import argparse
import logging
import nest_asyncio

from src.evals.experiment import run_eval_experiment

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run a Phoenix evaluation experiment.")
    parser.add_argument("--name", type=str, default="baseline")
    parser.add_argument("--description", type=str, default="Evaluation experiment")
    parser.add_argument("--dataset-name", type=str, default="meridian-policy-evals")

    args = parser.parse_args()

    logger.info("Running experiment: %s", args.name)
    experiment = run_eval_experiment(
        experiment_name=args.name,
        experiment_description=args.description,
        dataset_name=args.dataset_name,
    )
    logger.info("Done. Open http://localhost:6006 → Experiments to view results.")

if __name__ == "__main__":
    main()
