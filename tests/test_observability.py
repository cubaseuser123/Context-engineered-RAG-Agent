from contextlib import contextmanager

from src import observability


def test_span_uses_current_sdk_context_and_annotations(monkeypatch):
    events = []
    class Observation:
        def update(self, **kwargs):
            events.append(("update", kwargs))
    class Client:
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            events.append(("start", kwargs))
            yield Observation()
            events.append(("end",))
    monkeypatch.setattr(observability, "get_client", lambda: Client())
    with observability.span("retriever", query="hello") as annotate:
        annotate(count=3)
    assert events == [("start", {"name": "retriever", "as_type": "span", "metadata": {"query": "hello"}}), ("update", {"metadata": {"count": 3}}), ("end",)]
