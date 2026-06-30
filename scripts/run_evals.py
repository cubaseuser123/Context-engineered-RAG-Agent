"""
run_evals — Run all evaluators against traces already in Phoenix.
"""
import asyncio
import json
import logging
import sys

import nest_asyncio
import pandas as pd
from phoenix.client import Client
from phoenix.evals import async_evaluate_dataframe
from phoenix.evals.utils import to_annotation_dataframe

from src.config import PHOENIX_PROJECT_NAME, TEST_SUITE_PATH
from src.evals.extract import get_trace_dataframe, reshape_for_evals
from src.evals.evaluators import (
    eval_router_accuracy,
    eval_source_coverage,
    get_answer_correctness_evaluator,
    get_hallucination_evaluator,
    get_faithfulness_evaluator,
)
from src.tracing import init_tracing

# [TEACHING COMMENT]
# nest_asyncio lets us use 'async' and 'await' commands inside a standard python script.
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_suite() -> list | None:
    if not TEST_SUITE_PATH.exists():
        return None
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run_code_evaluators(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Run the two code-based evaluators on every row."""
    router_scores = []
    source_scores = []

    for _, row in eval_df.iterrows():
        row_dict = row.to_dict()
        router_result = eval_router_accuracy(row_dict)
        source_result = eval_source_coverage(row_dict)
        
        router_scores.append(router_result)
        source_scores.append(source_result)

    # [TEACHING COMMENT]
    # We extract the 'score' and 'label' from the dictionary results and put them into the Pandas table as new columns.
    router_accuracy_scores = []
    router_accuracy_labels = []
    for r in router_scores:
        router_accuracy_scores.append(r["score"])
        router_accuracy_labels.append(r["label"])
        
    eval_df["router_accuracy_score"] = router_accuracy_scores
    eval_df["router_accuracy_label"] = router_accuracy_labels
    
    source_coverage_scores = []
    source_coverage_labels = []
    for r in source_scores:
        source_coverage_scores.append(r["score"])
        source_coverage_labels.append(r["label"])

    eval_df["source_coverage_score"] = source_coverage_scores
    eval_df["source_coverage_label"] = source_coverage_labels

    return eval_df

async def run_llm_evaluators(eval_df: pd.DataFrame) -> None:
    """Run the three LLM-based evaluators using Phoenix's async batch evaluation."""
    client = Client()

    # Create a specific DataFrame for Answer Correctness
    correctness_df = eval_df[["span_id", "query", "response"]].copy()
    correctness_df = correctness_df.rename(columns={"span_id": "context.span_id"})

    if "expected_answer" in eval_df.columns:
        correctness_df["reference"] = eval_df["expected_answer"].fillna("")
        correctness_df["input"] = eval_df["query"]
        correctness_df["output"] = eval_df["response"]

        correctness_df = correctness_df[correctness_df["reference"] != ""]

        if not correctness_df.empty:
            logger.info("Running answer correctness evaluator...")
            correctness_evaluator = get_answer_correctness_evaluator()
            
            # [TEACHING COMMENT]
            # 'await' tells Python to pause and wait for the LLM to finish grading all rows.
            correctness_results = await async_evaluate_dataframe(
                dataframe=correctness_df,
                evaluators=[correctness_evaluator],
                concurrency=5,
            )

            annotations = to_annotation_dataframe(correctness_results)
            client.spans.log_span_annotations_dataframe(dataframe=annotations)

    # Create a specific DataFrame for Hallucination and Faithfulness
    grounding_df = eval_df[["span_id", "query", "response", "retrieved_chunks_text"]].copy()
    grounding_df = grounding_df.rename(columns={
        "span_id": "context.span_id",
        "retrieved_chunks_text": "reference",
    })
    grounding_df["input"] = eval_df["query"]
    grounding_df["output"] = eval_df["response"]

    grounding_df = grounding_df[grounding_df["reference"] != ""]

    if not grounding_df.empty:
        logger.info("Running hallucination + faithfulness evaluators...")
        hallucination_evaluator = get_hallucination_evaluator()
        faithfulness_evaluator = get_faithfulness_evaluator()

        grounding_results = await async_evaluate_dataframe(
            dataframe=grounding_df,
            evaluators=[hallucination_evaluator, faithfulness_evaluator],
            concurrency=5,
        )

        annotations = to_annotation_dataframe(grounding_results)
        client.spans.log_span_annotations_dataframe(dataframe=annotations)

def print_summary(eval_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total traces evaluated:   {len(eval_df)}")
    print("  LLM eval results logged as annotations in Phoenix.")
    print("  Open http://localhost:6006 → click any trace → see eval scores.")

async def main():
    init_tracing()
    spans_df = get_trace_dataframe()
    if spans_df.empty:
        sys.exit(1)

    test_suite = load_test_suite()
    eval_df = reshape_for_evals(spans_df, test_suite=test_suite)
    
    eval_df = run_code_evaluators(eval_df)
    await run_llm_evaluators(eval_df)
    print_summary(eval_df)
    logger.info("All evaluations complete.")

if __name__ == "__main__":
    asyncio.run(main())
