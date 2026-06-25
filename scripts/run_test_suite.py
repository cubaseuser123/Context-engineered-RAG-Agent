"""
run_test_suite - Execute the 30-question test suite and produce a scorecard.
"""
from os import PathLike
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path 
from src.config import TEST_SUITE_PATH, PROJECT_ROOT
from src.graph import compile_agent, run_query
from src.tracing import init_tracing

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

def load_test_suite() -> list:
    """Load test questions from corpus/test_suite.json."""
    if not TEST_SUITE_PATH.exists():
        logger.error("Test suite not found: %s", TEST_SUITE_PATH)
        sys.exit(1)
    
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def score_question(question: dict, result: dict) -> dict:
    """Score a single question against the agent's result."""
    actual_intent = result.get("intent", "")

    if actual_intent == question["expected_intent"]:
        router_correct = True
    else:
        router_correct = False 

    cited = result.get("cited_sources", [])
    expected = question.get("expected_sources", [])

    source_cited = False 
    if not expected:
        source_correct = False 
    if not expected:
        source_correct = True
    else:
        for exp in expected:
            if exp in cited:
                source_correct = True 

    #for hallucinations (or IDK questions)
    hallucinated = False 
    if "correct_answer_is_idk" in question and question["correct_answer_is_idk"]:
        response = result.get("response", "").lower()
        idk_phrases = ["i don't have", "not in the available", "i can't find", "no information"]

    has_idk = False 
    for phrase in idk_phrases:
        if phrase in response:
            has_idk = True 
    if not has_idk:
        hallucinated = True 
    
    budget_enforced = False 
    
    if "budget_log" in result:
        if "enforced" in result["budget_log"]:
            budget_enforced = result["budget_log"]["enforced"]
        
    return{
        "question_id" : question["id"],
        "category" : question["category"],
        "question" : question["question"],
        "expected_intent" : question["expected_intent"],
        "actual_intent" : actual_intent,
        "router_correct" : router_correct,
        "expected_sources" : expected,
        "actual_sources" : cited,
        "source_correct" : source_correct,
        "answer_correct" : None,
        "hallucinated" : hallucinated,
        "response_preview" : result.get("response", "")[:200],
        "budget_enforced" : budget_enforced,
    }

def compute_aggregate(results: list) -> dict:
    """Compute aggregate scorecard from individual question scores."""
    total = len(results)
    if total == 0:
        return {}
    
    router_correct = 0
    source_correct = 0
    hallucinated = 0
    budget_enforced = 0

    for r in results:
        if r["router_correct"]:
            router_correct += 1
        if r["source_correct"]:
            source_correct +=1 
        if r["hallucinated"]:
            hallucinated += 1
        if r["budget_enforced"]:
            budget_enforced += 1

    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "router_correct": 0, "source_correct": 0}
        
        categories[cat]["total"] += 1
        if r["router_correct"]:
            categories[cat]["router_correct"] += 1
        if r["source_correct"]:
            categories[cat]["source_correct"] += 1
        
    return {
        "total_questions": total,
        "router_accuracy": round(router_correct / total * 100, 1),
        "source_accuracy": round(source_correct / total * 100, 1),
        "hallucination_rate": round(hallucinated / total * 100, 1),
        "hallucination_count": hallucinated,
        "budget_enforcement_count": budget_enforced,
        "per_category": categories,
    }
    
def save_results(results: list, aggregate: dict) -> Path:
    """Save scorecard to a JSON file."""
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"scorecard_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "aggregate": aggregate,
        "questions": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path

def print_scorecard(aggregate: dict) -> None:
    """Print a human-readable scorecard to stdout."""
    print("\n" + "=" * 60)
    print("  SCORECARD")
    print("=" * 60)
    print(f"  Total questions:        {aggregate['total_questions']}")
    print(f"  Router accuracy:        {aggregate['router_accuracy']}%")
    print(f"  Source accuracy:         {aggregate['source_accuracy']}%")
    print(f"  Hallucination rate:     {aggregate['hallucination_rate']}%")
    print(f"  Budget enforcements:    {aggregate['budget_enforcement_count']}")
    print("-" * 60)
    print("  Per-category breakdown:")
    for cat, data in aggregate.get("per_category", {}).items():
        router_pct = round(data["router_correct"] / data["total"] * 100, 1)
        source_pct = round(data["source_correct"] / data["total"] * 100, 1)
        print(f"    {cat}: router={router_pct}%, source={source_pct}% (n={data['total']})")
    print("=" * 60 + "\n")

def main():
    init_tracing()
    questions = load_test_suite()
    logger.info("Loaded %d test questions", len(questions))

    app = compile_agent()

    results = []
    # [TEACHING COMMENT]
    # enumerate() gives us both the index 'i' and the item 'q' from the list.
    for i, q in enumerate(questions):
        logger.info("[%d/%d] Running: %s", i + 1, len(questions), q["question"][:60])
        try:
            result = run_query(app, query=q["question"], user_id="test_user", turn=i)
            scored = score_question(q, result)
            results.append(scored)
        except Exception as e:
            logger.error("Failed: %s", str(e))
            results.append({
                "question_id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "router_correct": False,
                "source_correct": False,
                "hallucinated": False,
                "answer_correct": None,
                "error": str(e),
            })

    aggregate = compute_aggregate(results)
    print_scorecard(aggregate)

    output_path = save_results(results, aggregate)
    logger.info("Scorecard saved to %s", output_path)

if __name__ == "__main__":
    main()
    

    