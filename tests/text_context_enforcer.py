"""
text_context_enforcer - Unit tests for context budget enforcer node.
"""
from src.nodes.context_enforcer import context_enforcer_node

def _make_chunk(doc: str, score: float, content_len: int = 100) -> dict: 
    """Create a fake chunk with controllable content length."""
    content_str = ""
    for _ in range(content_len):
        content_str += "x"
    
    return {
        "content" : content_str,
        "source_doc" : doc,
        "section" : "chunk_0",
        "department" : "HR",
        "score" : score,
    }

def _make_state(chunks=None, memory=None, history=None) -> dict:
    return {
        "retrieved_chunks": chunks or [],
        "memory_entries": memory or [],
        "conversation_history": history or []
    }

def test_under_budget_passthrough():
    """When total tokens are under budget, nothing should be trimmed"""
    state = _make_state(
        chunks = [_make_chunk("doc1", 0.9, content_len=50)],
        memory=[{"fact": "user is in HR", "timestamp": "2025-01-01", "source_turn": 1}],
        history= [{"role": "user", "content": "hello"}],
    )
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is False 
    assert len(result["trimmed_chunks"]) == 1
    assert len(result["trimmed_memory"]) == 1

def test_retrieval_trimming():
    """When retrieval exceeds budget, lowest-scoring chunks are passed through"""
    chunks = []
    for i in range(10):
        chunks.append(_make_chunk(f"doc{i}", score=i*0.1, content_len=800))
    state = _make_state(chunks = chunks)
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is True
    assert len(result["trimmed_chunks"]) < 10

def test_conversation_trimming():
    """When conversation history exceeds budget, oldest turns are dropped"""
    history = []
    for _ in range(20):
        content_str = ""
        for _ in range(100):
            content_str += "message "
        history.append({"role": "user", "content": content_str})
    
    state = _make_state(history=history)
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is True 
    assert len(result["trimmed_history"]) < 20
    
def test_memory_trimming():
    """When memory exceeds budget, oldest entries are dropped."""
    memory = []
    for i in range(20):
        fact_str = "fact"
        for _ in range(200):
            fact_str += "x"
        fact_str += f"  {i}"
        memory.append({"fact": fact_str, "timestamp": f"2025-01-{i+1:02d}", "source_turn": i})

    state = _make_state(memory=memory)
    result = context_enforcer_node(state)
    assert result["budget_log"]["enforced"] is True
    assert len(result["trimmed_memory"]) < 20

def test_budget_log_structure():
    """Budget log should always have the expected structure."""
    state = _make_state()
    result = context_enforcer_node(state)
    log = result["budget_log"]

    assert "zones" in log 
    assert "total_before" in log
    assert "total_after" in log 
    assert "enforced" in log 
    assert "drop_details" in log 

    assert len(log["zones"]) == 4
    zone_names = []
    for z in log["zones"]:
        zone_names.append(z["zone_name"])
    
    assert "system_prompt" in zone_names
    assert "retrieval" in zone_names 
    assert "memory" in zone_names
    assert "conversation" in zone_names 