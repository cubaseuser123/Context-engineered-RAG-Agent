"""
test_memory_store - Unit tests for the SQLite memory store.
"""
from src.stores import memory_store 

TEST_USER = "test_user_unit"

def setup_function():
    """Before each test clear the previous user"""
    memory_store.clear(TEST_USER)

def test_write_and_read():
    """Write facts and read them back"""
    memory_store.write(TEST_USER, ["fact one", "fact two"], turn=1)
    entries = memory_store.read(TEST_USER)
    assert len(entries) == 2

    facts = []
    for e in entries:
        facts.append(e["fact"])
    
    assert "fact one" in facts 
    assert "fact two" in facts 

def test_read_empty():
    entries = memory_store.read("nonexistent_user_xyz")
    assert entries == []

def test_write_empty_list():
    count = memory_store.write(TEST_USER, [], turn = 1)
    assert count == 0

def test_prune():
    """Remove oldest entries beyond cap"""
    for i in range(10):
        memory_store.write(TEST_USER, [f"fact {i}"], turn=i)
    deleted = memory_store.prune(TEST_USER, max_entries=3)
    assert deleted == 7
    remaining = memory_store.read(TEST_USER)
    assert len(remaining) == 3

def test_clear():
    memory_store.write(TEST_USER, ["a", "b", "c"], turn = 1)
    deleted = memory_store.clear(TEST_USER)
    assert deleted == 3
    entries = memory_store.read(TEST_USER)
    assert entries == []

def test_ordering():
    memory_store.write(TEST_USER, ["old fact"], turn = 1)
    memory_store.write(TEST_USER, ["new fact"], turn = 2)
    entries = memory_store.read(TEST_USER)
    assert entries[0]["fact"] == "new fact"
