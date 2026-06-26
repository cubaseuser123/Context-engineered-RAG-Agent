"""
evaluators - Five evaluators for the context-engineered RAG agent.
"""
import logging 
from phoenix.evals import ClassificationEvaluator, LLM
from src.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

_judge_llm = None

def _get_judge() -> LLM:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = LLM(provider="google", model="gemini-2.5-pro")
    return _judge_llm

def eval_router_accuracy(row : dict) -> dict:
    """
    Code evaluator : compare agent's intent to expected intent.
    """
    actual = row.get("intent", "")
    expected = row.get("expected_intent", "")

    if not expected:
        return {"score" : None, "label": "no_ground_truth", "explanation" : "No expected_intent in test suite"}
    
    if actual == expected:
        return {"score" : 1.0, "label": "correct", "explanation" : f"Expected '{expected}', got '{actual}'"}
    else:
        return {"score" : 0.0, "label": "incorrect", "explanation" : f"Expected '{expected}', got '{actual}'"}

def eval_source_coverage(row: dict) -> dict:
    """
    Code evaluator: check if cited sources overlap with expected sources.
    """
    cited = row.get("cited_sources", [])
    expected = row.get("expected_sources", [])

    if not expected:
        return{"score" : None, "label": "no_ground_truth", "explanation" : "No expected_sources in test suite"}

    if not cited:
        return{"score" : 0.0, "label" : "no_citations", "explanation" : f"Agent cited nothing, expected {expected}"}

    cited_lower = set()
    for s in cited:
        cited_lower.add(s.lower().strip())
    
    expected_lower = set()
    for s in expected:
        if isinstance(s, str):
            expected_lower.add(s.lower().strip())
    
    found = set()
    for exp in expected_lower:
        for cit in cited_lower:
            if exp in cit or cit in exp:
                found.add(exp)
    
    if len(found) > 0:
        coverage = len(found) / len(expected_lower)
        if coverage == 1.0:
            label = "covered"
        else:
            label = "partial"
        return {"score" : coverage, "label" : label, "explanation" : f"Found {len(found)}/{len(expected_lower)} expected sources."}

    return {"score" : 0.0, "label" : "missing", "explanation": f"No overlap between expected and cited sources."}

ANSWER_CORRECTNESS_TEMPLATE = """You are a strict grader evaluating a policy assistant's response.

You will be given:
- The user's question
- The assistant's response
- The expected correct answer

Your job is to determine if the assistant's response is CORRECT — meaning it conveys the same factual information as the expected answer, even if the wording is different.

Rules:
- "correct" means the response contains the key facts from the expected answer
- Minor wording differences are OK (e.g., "12 days" vs "twelve days")
- The response may contain additional correct context — that's fine, not a penalty
- "incorrect" means the response is missing key facts, states wrong facts, or contradicts the expected answer
- If the expected answer says the correct response is "I don't know" and the assistant says it doesn't have the information, that is CORRECT

[BEGIN DATA]
************
[Question]: {input}
************
[Assistant Response]: {output}
************
[Expected Answer]: {reference}
************
[END DATA]

Is the assistant's response correct or incorrect based on the expected answer?"""

def get_answer_correctness_evaluator() -> ClassificationEvaluator:
    return ClassificationEvaluator(
        name = "answer_correctness",
        llm = _get_judge(),
        prompt_template = ANSWER_CORRECTNESS_TEMPLATE,
        choices = {"correct" : 1.0, "incorrect" : 0.0},
    )

HALLUCINATION_TEMPLATE = """You are evaluating whether an AI assistant's response contains hallucinated information.

A "hallucination" means the response states facts, details, or claims that are NOT present in the provided reference context. The reference context is the set of documents the assistant was given to answer the question.

You will be given:
- The user's question
- The assistant's response
- The reference context (retrieved documents the assistant had access to)

Rules:
- "factual" means every claim in the response can be traced back to the reference context
- "hallucinated" means the response contains at least one claim that cannot be found in the reference context
- If the response says "I don't have that information" and the reference context indeed doesn't contain the answer, that is "factual"
- Invented phone numbers, email addresses, URLs, or specific figures not in the context are hallucinations
- General knowledge statements (like "please contact HR") are acceptable if the context mentions HR

[BEGIN DATA]
************
[Question]: {input}
************
[Assistant Response]: {output}
************
[Reference Context]: {reference}
************
[END DATA]

Is the assistant's response factual or hallucinated based on the reference context?"""

def get_hallucination_evaluator() -> ClassificationEvaluator:
    return ClassificationEvaluator(
        name = "hallucination",
        llm = _get_judge(),
        prompt_template=HALLUCINATION_TEMPLATE,
        choices={"factual": 1.0, "hallucinated": 0.0},
    )

FAITHFULNESS_TEMPLATE = """You are evaluating whether an AI policy assistant's response is faithful to the source documents it was given.

"Faithful" means the response ONLY uses information present in the reference context. It does not add external knowledge, personal opinions, or information from sources not provided.

You will be given:
- The user's question
- The assistant's response
- The reference context (the policy documents the assistant was given)

Rate the faithfulness:
- "faithful": Every statement in the response is supported by the reference context
- "partially_faithful": Most of the response is supported, but some minor details come from outside the context
- "unfaithful": The response significantly departs from or goes beyond the reference context

[BEGIN DATA]
************
[Question]: {input}
************
[Assistant Response]: {output}
************
[Reference Context]: {reference}
************
[END DATA]

Is the assistant's response faithful, partially_faithful, or unfaithful?"""

def get_faithfulness_evaluator() -> ClassificationEvaluator:
    return ClassificationEvaluator(
        name = "faithfulness",
        llm=_get_judge(),
        prompt_template=FAITHFULNESS_TEMPLATE,
        choices={"faithful": 1.0, "partially_faithful": 0.5, "unfaithful":0.0},
    )