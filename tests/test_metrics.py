from evals.dataset import Expected, Scenario, Turn
from evals.metrics import check_budget_invariant, evaluate_turn, refusal_confusion
from src.models.run_record import (
    CostInfo,
    OutputInfo,
    RetrievalInfo,
    RetrievedChunk,
    RoutingInfo,
    RunRecord,
)

BUDGET = {
    "zones": [
        {"zone_name": "retrieval", "cap": 3000, "tokens_after": 900, "dropped": []},
        {"zone_name": "memory", "cap": 500, "tokens_after": 0, "dropped": []},
        {"zone_name": "conversation", "cap": 1000, "tokens_after": 0, "dropped": []},
    ],
    "total_after": 1100,
    "total_cap": 5000,
    "enforced": False,
}


def _record(**overrides) -> RunRecord:
    payload = dict(
        run_id="run",
        record_id="s:00",
        scenario_id="s",
        turn_index=0,
        arm="engineered",
        config={},
        question="How many leave days can I carry forward?",
        retrieval=RetrievalInfo(
            query="carry forward",
            k=8,
            searches=1,
            chunk_ids=["c1", "c2"],
            retained_chunk_ids=["c1", "c2"],
            chunks=[
                RetrievedChunk(chunk_id="c1", doc="HR_Leave_Policy", score=0.91),
                RetrievedChunk(chunk_id="c2", doc="HR_Parental_Leave", score=0.80),
            ],
        ),
        output=OutputInfo(
            response="You may carry forward 12 days [Source: HR_Leave_Policy].",
            citations=["HR_Leave_Policy"],
        ),
        cost=CostInfo(
            model="qwen/qwen3.8-27b",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
        ),
        duration_ms=42.0,
        budget=BUDGET,
        routing=RoutingInfo(intent="policy_lookup"),
    )
    payload.update(overrides)
    return RunRecord(**payload)


def _scenario(**overrides) -> Scenario:
    payload = dict(
        id="s",
        cohort="single_doc",
        turns=[Turn(content="How many leave days can I carry forward?")],
        expected=Expected(
            intent="policy_lookup",
            docs=["HR_Leave_Policy"],
            must_include=["12 days"],
            reference_answer="12 days may be carried forward.",
        ),
    )
    payload.update(overrides)
    return Scenario(**payload)


def test_code_evaluators_score_a_good_turn():
    evaluation = evaluate_turn(_record(), _scenario(), 0)
    assert evaluation.metrics["routing_accuracy"] is True
    assert evaluation.metrics["retrieval_recall_at_k"] == 1.0
    assert evaluation.metrics["retrieval_precision_at_k"] == 0.5
    assert evaluation.metrics["citation_precision"] == 1.0
    assert evaluation.metrics["citation_recall"] == 1.0
    assert evaluation.metrics["constraint_pass"] is True
    assert evaluation.metrics["budget_invariant"] is True
    assert evaluation.metrics["tokens_per_answer"] == 110.0


def test_missing_required_value_fails_the_constraint():
    record = _record(output=OutputInfo(response="It depends.", citations=[]))
    evaluation = evaluate_turn(record, _scenario(), 0)
    assert evaluation.metrics["constraint_pass"] is False
    # answering with no citation at all is a precision failure, not a pass
    assert evaluation.metrics["citation_precision"] == 0.0


def test_expected_document_missing_from_retrieval_is_a_recall_miss():
    record = _record(
        retrieval=RetrievalInfo(
            query="q",
            k=8,
            searches=1,
            chunk_ids=["c2"],
            retained_chunk_ids=["c2"],
            chunks=[RetrievedChunk(chunk_id="c2", doc="HR_Parental_Leave", score=0.5)],
        )
    )
    evaluation = evaluate_turn(record, _scenario(), 0)
    assert evaluation.metrics["retrieval_recall_at_k"] == 0.0


def test_budget_invariant_checks_caps_and_drop_order():
    assert check_budget_invariant(BUDGET, [], ["c1"]) is True

    over_cap = {
        "zones": [{"zone_name": "retrieval", "cap": 5, "tokens_after": 10, "dropped": []}],
        "total_after": 10,
        "total_cap": 5,
    }
    assert check_budget_invariant(over_cap, [], []) is False

    bad_drop = {
        "zones": [
            {
                "zone_name": "retrieval",
                "cap": 3000,
                "tokens_after": 100,
                "dropped": [
                    {
                        "zone": "retrieval",
                        "item_id": "c9",
                        "tokens": 10,
                        "reason": "lowest_score",
                        "score": 0.99,
                    }
                ],
            }
        ],
        "total_after": 100,
        "total_cap": 5000,
    }
    retrieved = [{"chunk_id": "c1", "score": 0.1}, {"chunk_id": "c9", "score": 0.99}]
    assert check_budget_invariant(bad_drop, retrieved, ["c1"]) is False


def test_refusal_confusion_counts_the_four_cells():
    evaluations = [
        evaluate_turn(
            _record(output=OutputInfo(response="I'd rather not.", refused=True)),
            _scenario(cohort="refusal", expected=Expected(refusal_warranted=True)),
            0,
        ),
        evaluate_turn(
            _record(output=OutputInfo(response="Answer.", refused=False)),
            _scenario(cohort="refusal", expected=Expected(refusal_warranted=True)),
            0,
        ),
        evaluate_turn(
            _record(output=OutputInfo(response="Refusal.", refused=True)),
            _scenario(expected=Expected(refusal_warranted=False)),
            0,
        ),
    ]
    confusion = refusal_confusion(evaluations)
    assert confusion["refused_warranted"] == 1
    assert confusion["answered_warranted"] == 1
    assert confusion["refused_unwarranted"] == 1
