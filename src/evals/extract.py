"""
extract - Pull traces from phoenix and reshape them into eval-ready DataFrames.
"""
import logging 
import pandas as pd 
from phoenix.client import Client 

from src.config import PHOENIX_PROJECT_NAME

logger = logging.getLogger(__name__)

def get_trace_dataframe(project_name: str = PHOENIX_PROJECT_NAME) -> pd.DataFrame:
    """
    Export all top-level spans from Phoenix as a DataFrame.
    """
    client = Client()

    spans_df = client.spans.get_spans_dataframe(project_name = project_name)

    if spans_df is None or spans_df.empty:
        logger.warning("No spans found in Phoenix project '%s'", project_name)
        return pd.DataFrame()

    logger.info("Exported %d spans from Phoenix project '%s'", len(spans_df), project_name)
    return spans_df

def reshape_for_evals(spans_df: pd.DataFrame, test_suite: list = None) -> pd.DataFrame:
    """
    Reshape raw phoenix spans into a flat DataFrame for evals.
    """
    if spans_df.empty:
        return pd.DataFrame()
    
    rows = []

    for index, span in spans_df.iterrows():
        input_val = span.get("attributes.input.value")
        output_val = span.get("attributes.output.value")

        if not isinstance(input_val, dict) or not isinstance(output_val, dict):
            continue 

        query = input_val.get("query", "")
        if not query:
            continue

        row = {
            "span_id" : span.get("context.span_id", ""),
            "trace_id" : span.get("context.trace_id", ""),
            "query" : query,
            "response" : output_val.get("response", ""),
            "intent" : output_val.get("intent", ""),
            "cited_sources": output_val.get("cited_sources", []),
            "retrieved_chunks_text": _extract_chunk_text(output_val),
            "budget_enforced" : output_val.get("budget_log", {}).get("enforced", False),
        }
        rows.append(row)

    eval_df = pd.DataFrame(rows)

    if test_suite and not eval_df.empty:
        gt_df = pd.DataFrame(test_suite)
        gt_df = gt_df.rename(columns={"question" : "query"})
        gt_cols = ["query", "expected_intent", "expected_sources", "expected_answer", "correct_answer_is_idk"]

        valid_cols = []
        for c in gt_cols:
            if c in gt_df.columns:
                valid_cols.append(c)
        gt_df = gt_df[valid_cols]

        eval_df = eval_df.merge(gt_df, on="query", how="left")

    return eval_df

def _extract_chunk_text(output: dict) -> str:
    """
    Concatenate all retrieved chunk contents into a single string.
    """
    chunks = output.get("trimmed_chunks", output.get("retrieved_chunks", []))
    if not chunks:
        return ""
    
    chunk_strings = []
    for c in chunks:
        if isinstance(c, dict):
            source = c.get("source_doc", "unknown")
            content = c.get("content", "")
            chunk_strings.append(f"[Source: {source}]\n{content}")
    return "\n\n".join(chunk_strings)

    