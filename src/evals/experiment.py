"""
experiment — Phoenix experiment runner.
"""
import json
import logging
import pandas as pd

from phoenix.client import Client
from phoenix.client.experiments import create_evaluator

from src.config import TEST_SUITE_PATH, PHOENIX_PROJECT_NAME
from src.graph import compile_agent, run_query
from src.tracing import init_tracing

logger = logging.getLogger(__name__)

def load_test_suite() -> list[dict]:
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def upload_dataset(client: Client, test_suite: list[dict], dataset_name: str = "meridian-policy-evals"):
    """Upload the test suite as a Phoenix dataset."""
    rows = []
    for q in test_suite:
        # Check if correct_answer_is_idk is in the dictionary, otherwise default to False
        is_idk = False
        if "correct_answer_is_idk" in q:
            is_idk = q["correct_answer_is_idk"]
            
        rows.append({
            "input.query": q["question"],
            "input.user_id": "eval_user",
            "output.expected_intent": q["expected_intent"],
            "output.expected_sources": json.dumps(q["expected_sources"]),
            "output.expected_answer": q["expected_answer"],
            "metadata.category": q["category"],
            "metadata.correct_answer_is_idk": is_idk,
            "metadata.question_id": q["id"],
        })

    df = pd.DataFrame(rows)

    dataset = client.datasets.create_dataset(
        dataframe=df,
        name=dataset_name,
        input_keys=["input.query", "input.user_id"],
        output_keys=["output.expected_intent", "output.expected_sources", "output.expected_answer"],
        metadata_keys=["metadata.category", "metadata.correct_answer_is_idk", "metadata.question_id"],
    )
    return dataset

def create_agent_task():
    """Create the task function that Phoenix calls for each test case."""
    app = compile_agent()

    def agent_task(example: dict) -> dict:
        input_data = example.get("input", {})
        query = input_data.get("query", "")
        user_id = input_data.get("user_id", "eval_user")

        result = run_query(app, query=query, user_id=user_id, turn=0)

        chunks = result.get("trimmed_chunks", result.get("retrieved_chunks", []))
        
        chunk_strings = []
        for c in chunks:
            if isinstance(c, dict):
                source = c.get('source_doc', 'unknown')
                content = c.get('content', '')
                chunk_strings.append(f"[Source: {source}]\n{content}")
                
        context_text = "\n\n".join(chunk_strings)

        # Check if budget_log exists and if enforced is true
        budget_enforced = False
        if "budget_log" in result:
            budget_log = result["budget_log"]
            if "enforced" in budget_log:
                budget_enforced = budget_log["enforced"]

        return {
            "response": result.get("response", ""),
            "intent": result.get("intent", ""),
            "cited_sources": result.get("cited_sources", []),
            "retrieved_context": context_text,
            "budget_enforced": budget_enforced,
        }

    return agent_task

def build_experiment_evaluators() -> list:
    """Build evaluators in the format Phoenix experiments expect."""
    
    # [TEACHING COMMENT]
    # @create_evaluator is a decorator required by Phoenix to register these functions as evaluators for experiments.
    @create_evaluator(kind="code", name="router_accuracy")
    def router_eval(input: dict, output: dict, expected: dict) -> float:
        actual_intent = output.get("intent", "")
        expected_intent = expected.get("expected_intent", "")
        if actual_intent == expected_intent:
            return 1.0
        else:
            return 0.0

    @create_evaluator(kind="code", name="source_coverage")
    def source_eval(input: dict, output: dict, expected: dict) -> float:
        cited = output.get("cited_sources", [])
        expected_raw = expected.get("expected_sources", "[]")
        
        if isinstance(expected_raw, str):
            expected_sources = json.loads(expected_raw)
        else:
            expected_sources = expected_raw

        if not expected_sources:
            return 1.0

        cited_lower = set()
        for s in cited:
            if isinstance(s, str):
                cited_lower.add(s.lower().strip())
                
        expected_lower = set()
        for s in expected_sources:
            if isinstance(s, str):
                expected_lower.add(s.lower().strip())

        for exp in expected_lower:
            for cit in cited_lower:
                if exp in cit or cit in exp:
                    return 1.0
        return 0.0

    @create_evaluator(kind="code", name="hallucination_check")
    def hallucination_code_eval(input: dict, output: dict, expected: dict) -> float:
        is_idk = False
        if "correct_answer_is_idk" in expected:
            is_idk = expected["correct_answer_is_idk"]
            
        if isinstance(is_idk, str):
            is_idk = is_idk.lower() == "true"
            
        if not is_idk:
            return 1.0 

        response = output.get("response", "").lower()
        idk_phrases = ["i don't have", "not in the available", "i can't find", "no information", "i don't know"]
        
        has_idk = False
        for phrase in idk_phrases:
            if phrase in response:
                has_idk = True
                
        if has_idk:
            return 1.0
        else:
            return 0.0

    return [router_eval, source_eval, hallucination_code_eval]

def run_eval_experiment(experiment_name: str = "baseline", experiment_description: str = "Baseline", dataset_name: str = "meridian-policy-evals"):
    init_tracing()
    client = Client()

    test_suite = load_test_suite()
    dataset = upload_dataset(client, test_suite, dataset_name=dataset_name)

    task = create_agent_task()
    evaluators = build_experiment_evaluators()

    logger.info("Starting experiment '%s' with %d questions", experiment_name, len(test_suite))

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description=experiment_description,
    )
    return experiment
