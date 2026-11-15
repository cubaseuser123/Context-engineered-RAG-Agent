"""dataset_to_langfuse.py - Register benchmark dataset in Langfuse Cloud.
"""
from __future__ import annotations

import logging
from dotenv import load_dotenv

load_dotenv()

from evals.dataset import load_dataset
from src.config import SCENARIOS_PATH
from src.tracing import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dataset_to_langfuse")


def upload_dataset():
    client = get_client()
    if not client:
        logger.error("Langfuse client not initialized. Check .env")
        return

    ds = load_dataset(SCENARIOS_PATH)
    dataset_name = f"scenarios.{ds.version}"

    logger.info("Creating dataset '%s' in Langfuse...", dataset_name)
    try:
        langfuse_ds = client.create_dataset(
            name=dataset_name,
            description="Meridian Technologies RAG Policy Benchmark (30 scenarios, 52 turns)",
            metadata={"version": ds.version, "scenarios_count": len(ds.scenarios)},
        )
    except Exception as e:
        logger.info("Dataset may already exist (%s), fetching existing...", e)
        langfuse_ds = client.get_dataset(dataset_name)

    items_created = 0
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
                items_created += 1
            except Exception as e:
                pass

    client.flush()
    logger.info("Successfully registered dataset with %d items in Langfuse!", items_created)


if __name__ == "__main__":
    upload_dataset()
