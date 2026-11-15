import json

from src.graph import route_by_intent
from src.nodes import router, memory_writer, retriever
from src.run_config import RunConfig
from src.llm import LLMResponse, Usage
from evals.judges import _prompt
from src.models.run_record import RunRecord


def test_followup_can_require_retrieval():
    assert route_by_intent({"intent": "clarification", "needs_retrieval": True}) == "retriever"
    assert route_by_intent({"intent": "clarification", "needs_retrieval": False}) == "context_enforcer"


def test_router_keeps_separate_search_and_memory_decisions(monkeypatch):
    payload = {"intent": "clarification", "needs_retrieval": True,
               "search_query": "IT vendor security review duration", "user_facts": [], "department_filter": "IT"}
    monkeypatch.setattr(router.llm, "chat", lambda *a, **k: LLMResponse(text=json.dumps(payload), model="stub", usage=Usage(), latency_ms=0))
    result = router.router_node({"query": "How long does that review take?"})
    assert result["search_query"] == payload["search_query"]
    assert result["needs_retrieval"] is True


def test_memory_uses_router_facts_without_another_model_call(monkeypatch):
    saved = []
    monkeypatch.setattr(memory_writer.memory_store, "read", lambda _: [])
    monkeypatch.setattr(memory_writer.memory_store, "write", lambda user, facts, turn: saved.extend(facts))
    monkeypatch.setattr(memory_writer.memory_store, "prune", lambda _: None)
    monkeypatch.setattr(router.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("extra model call")))
    result = memory_writer.memory_writer_node({"user_facts": ["On-call SRE for TRITON this week"], "user_id": "test"})
    assert saved == ["On-call SRE for TRITON this week"]
    assert result["new_memory_entries"]


def test_retrieval_uses_standalone_question_and_deduplicates(monkeypatch):
    queries = []
    def query(**kwargs):
        queries.append(kwargs["text"])
        return [{"chunk_id": str(i), "content": "Vendor review takes 30 days", "score": 0.8, "source_doc": "IT_Review"} for i in range(2)]
    monkeypatch.setattr(retriever.vector_store, "query", query)
    result = retriever.retriever_node({"query": "How long?", "search_query": "vendor review duration", "run_config": RunConfig.engineered().as_dict()})
    assert queries == ["vendor review duration"]
    assert len(result["retrieved_chunks"]) == 1


def test_judges_receive_evidence_beyond_old_cutoff():
    record = RunRecord(run_id="r", record_id="s:00", scenario_id="s", turn_index=0, arm="naive", prompt="x" * 6500 + "Important exception: 30 April")
    assert "Important exception: 30 April" in _prompt("groundedness", record, None)
    assert "Important exception: 30 April" in _prompt("citation_support", record, None)


def test_user_fact_statement_is_acknowledged_without_generation(monkeypatch):
    from src.nodes.synthesizer import synthesizer_node
    monkeypatch.setattr(router.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("extra call")))
    result = synthesizer_node({"intent": "memory_recall", "needs_retrieval": False,
                               "user_facts": ["I am on call this week"]})
    assert "on call this week" in result["response"]


def test_exact_duplicates_do_not_consume_the_evidence_budget(monkeypatch):
    chunks = [{"chunk_id": "a", "content": "Annual leave deadline", "source_doc": "HR", "score": .9},
              {"chunk_id": "b", "content": "Annual  leave deadline", "source_doc": "HR", "score": .8},
              {"chunk_id": "c", "content": "Exception: long tenured staff have a grace period", "source_doc": "HR", "score": .85}]
    monkeypatch.setattr(retriever.vector_store, "query", lambda **_: chunks)
    result = retriever.retriever_node({"query": "exception for tenured staff", "run_config": RunConfig.engineered().as_dict()})
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][0]["chunk_id"] == "c"
    assert result["retrieved_chunks"][0]["vector_score"] == .85


def test_router_receives_stored_facts_before_search(monkeypatch):
    import src.graph as graph
    seen = []
    monkeypatch.setattr(graph, "memory_reader_node", lambda _: {"memory_entries": [{"fact": "I joined in 2018"}]})
    def classify(state):
        seen.append(state["memory_entries"])
        return {"intent": "policy_lookup", "needs_retrieval": True}
    monkeypatch.setattr(graph, "router_node", classify)
    monkeypatch.setattr(graph, "retriever_node", lambda _: {"retrieved_chunks": []})
    monkeypatch.setattr(graph, "synthesizer_node", lambda _: {"response": "stub"})
    monkeypatch.setattr(graph, "memory_writer_node", lambda _: {})
    graph.build_graph(RunConfig.engineered()).compile().invoke({"query": "Does the grace period apply to me?"})
    assert seen == [[{"fact": "I joined in 2018"}]]


def test_router_prompt_contains_bounded_memory(monkeypatch):
    messages = []
    def chat(prompt, **kwargs):
        messages.extend(prompt)
        return LLMResponse(text=json.dumps({"intent": "policy_lookup"}), model="stub", usage=Usage(), latency_ms=0)
    monkeypatch.setattr(router.llm, "chat", chat)
    router.router_node({"query": "Does the grace period apply to me?",
                        "memory_entries": [{"fact": "I joined in 2018"}, {"fact": "irrelevant " * 1000}]})
    assert "I joined in 2018" in messages[1]["content"]
    assert "irrelevant " not in messages[1]["content"]
