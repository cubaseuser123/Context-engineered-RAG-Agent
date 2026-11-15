from src.documents import chunk_document, corpus_doc_ids, normalize_doc_id, parse_document
from src.token_utils import count_tokens


def test_normalize_strips_extension_directory_and_spaces():
    assert normalize_doc_id("HR_Leave_Policy.txt") == "HR_Leave_Policy"
    assert normalize_doc_id(r"corpus\documents\HR_Leave_Policy.txt") == "HR_Leave_Policy"
    assert normalize_doc_id("  Finance_Travel.txt ") == "Finance_Travel"


def test_parse_document_reads_the_metadata_header(tmp_path):
    path = tmp_path / "HR_Leave_Policy.txt"
    path.write_text(
        "Document: HR_Leave_Policy\nDepartment: HR\nVersion: 4.2\n---\nBody text\n",
        encoding="utf-8",
    )
    doc = parse_document(path)
    assert doc["doc_id"] == "HR_Leave_Policy"
    assert doc["department"] == "HR"
    assert doc["version"] == "4.2"
    assert doc["body"] == "Body text"


def test_chunk_ids_are_contiguous_and_deterministic():
    body = "Meridian policy clause. " * 400
    first = chunk_document("HR_Leave_Policy", body, chunk_size=100, overlap=20)
    second = chunk_document("HR_Leave_Policy", body, chunk_size=100, overlap=20)

    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]
    assert first[0]["chunk_id"] == "HR_Leave_Policy::000"
    assert first[1]["chunk_id"] == "HR_Leave_Policy::001"
    assert all(count_tokens(c["content"]) <= 100 for c in first)


def test_empty_body_produces_no_chunks():
    assert chunk_document("X", "   ") == []


def test_corpus_doc_ids_is_cached_and_sorted(tmp_path):
    (tmp_path / "B.txt").write_text("b", encoding="utf-8")
    (tmp_path / "A.txt").write_text("a", encoding="utf-8")
    assert corpus_doc_ids(tmp_path) == ("A", "B")
