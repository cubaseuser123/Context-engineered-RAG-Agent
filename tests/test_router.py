"""
test_router - Smoke tests for the router intent classifier.
These call the real Gemini API.
"""
import pytest 
from src.nodes.router import router_node

def _make_state(query: str, history: list = None) -> dict:
    if history is None:
        history = []
    return {"query" : query, "conversation_history": history, "user_id" : "test"}

@pytest.mark.parametrize("query, expected_intent", [
    ("What is the leave policy?", "policy_lookup"),
    ("How many sick days do I get?", "policy_lookup"),
    ("What's the weather like?", "out_of_scope"),
    ("Tell me a joke", "out_of_scope"),
])

def test_basic_routing(query, expected_intent):
    result = router_node(_make_state(query))
    assert result["intent"] == expected_intent

def test_router_returns_required_keys():
    result = router_node(_make_state("What is the expense policy?"))
    assert "intent" in result 
    assert "router_reasoning" in result 
    assert "retrieval_metadata_filter" in result 

def test_clarification_intent():
    history = [
        {"role" : "user", "content": "What is the travel policy?"},
        {"role": "assistant", "content": "The travel policy covers booking class and per diem rates."},
    ]
    result = router_node(_make_state("Can you explain that in more detail?", history))
    assert result["intent"] == "clarification"