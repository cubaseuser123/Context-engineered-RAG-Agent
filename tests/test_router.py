import json

import pytest

from src.errors import RouterParseError
from src.llm import LLMResponse, Usage
from src.nodes import router as router_module
from src.nodes.router import router_node
from src.run_config import RunConfig

CONFIG = RunConfig.engineered().as_dict()


def _response(payload: dict) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(payload),
        model="stub",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        latency_ms=1.0,
    )


@pytest.fixture
def stub_chat(monkeypatch):
    def fake_chat(messages, **kwargs):
        return _response(
            {
                "intent": "policy_lookup",
                "reasoning": "asks about leave",
                "department_filter": "HR",
            }
        )

    monkeypatch.setattr(router_module.llm, "chat", fake_chat)


def test_router_writes_the_key_the_retriever_reads(stub_chat):
    result = router_node(
        {
            "query": "How many leave days can I carry forward?",
            "conversation_history": [],
            "run_config": CONFIG,
        }
    )

    assert result["intent"] == "policy_lookup"
    # The defect the rewrite fixes: the retriever reads
    # retrieval_metadata_filter, so the router must write that exact key.
    assert result["retrieval_metadata_filter"] == "HR"
    assert result["department_filter_requested"] == "HR"
    assert result["router_usage"]["total_tokens"] == 15


def test_router_rejects_unknown_department(monkeypatch):
    monkeypatch.setattr(
        router_module.llm,
        "chat",
        lambda messages, **kwargs: _response(
            {"intent": "policy_lookup", "reasoning": "x", "department_filter": "Marketing"}
        ),
    )
    result = router_node({"query": "hello", "run_config": CONFIG})
    assert result["retrieval_metadata_filter"] is None


def test_router_falls_back_to_out_of_scope_on_unknown_intent(monkeypatch):
    monkeypatch.setattr(
        router_module.llm,
        "chat",
        lambda messages, **kwargs: _response(
            {"intent": "chitchat", "reasoning": "x", "department_filter": None}
        ),
    )
    result = router_node({"query": "hi", "run_config": CONFIG})
    assert result["intent"] == "out_of_scope"


def test_router_raises_on_non_json(monkeypatch):
    monkeypatch.setattr(
        router_module.llm,
        "chat",
        lambda messages, **kwargs: LLMResponse(
            text="Sure! Here is my classification:", model="stub", usage=Usage(), latency_ms=1.0
        ),
    )
    with pytest.raises(RouterParseError):
        router_node({"query": "hi", "run_config": CONFIG})
