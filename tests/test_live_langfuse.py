from types import SimpleNamespace
import pytest
from scripts.live_langfuse import make_invoker
from scripts.score_langfuse_records import publish_resource_metrics


def test_turn_executes_inside_experiment_and_returns_original_state():
    events = []
    item = SimpleNamespace(id="case:00")
    state = {"response": "answer", "retrieved_chunks": ["evidence"]}
    class Client:
        def get_dataset(self, name):
            return SimpleNamespace(items=[item])
        def run_experiment(self, **kwargs):
            events.append("experiment")
            assert kwargs["run_name"] == "run-naive"
            assert kwargs["data"] == [item]
            assert kwargs["task"](item=item) == {"response": "answer"}
            return SimpleNamespace(item_results=[SimpleNamespace(trace_id="live-trace")])
    class App:
        def invoke(self, value):
            events.append("model")
            return state
    result, trace = make_invoker(Client(), "dataset", "run-naive", "naive")(App(), {}, "case", 0)
    assert events == ["experiment", "model"]
    assert result is state
    assert trace == "live-trace"


def test_failed_experiment_is_not_reported_as_success():
    item = SimpleNamespace(id="case:00")
    client = SimpleNamespace(get_dataset=lambda _: SimpleNamespace(items=[item]),
                             run_experiment=lambda **_: SimpleNamespace(item_results=[]))
    with pytest.raises(RuntimeError, match="Experiment turn failed"):
        make_invoker(client, "dataset", "run", "naive")(None, {}, "case", 0)


def test_resource_scores_include_auxiliary_calls_and_keep_stable_ids():
    calls = []
    client = SimpleNamespace(create_score=lambda **kwargs: calls.append(kwargs))
    record = {"trace_id": "trace", "cost": {"total_tokens": 100, "aux_prompt_tokens": 20, "aux_completion_tokens": 5},
              "duration_ms": 42, "output": {"refused": False}, "error": None}
    publish_resource_metrics(client, record)
    publish_resource_metrics(client, record)
    assert next(x["value"] for x in calls if x["name"] == "total_agent_tokens") == 125
    assert [x["score_id"] for x in calls[:5]] == [x["score_id"] for x in calls[5:]]
