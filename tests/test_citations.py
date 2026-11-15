from src.citations import doc_aliases, extract_citations

CANDIDATES = [
    "HR_Leave_Policy",
    "HR_Parental_Leave",
    "Finance_Travel_And_Expense_Policy",
    "Operations_Sev_Incident_Escalation",
]


def test_reads_source_markers():
    result = extract_citations(
        "According to [Source: HR_Leave_Policy], 12 days may be carried forward.",
        CANDIDATES,
    )
    assert result.cited == ["HR_Leave_Policy"]
    assert result.unknown == []
    assert result.markers == ["HR_Leave_Policy"]


def test_resolves_human_document_names():
    result = extract_citations(
        "According to the Leave Policy, you may carry forward 12 days.", CANDIDATES
    )
    assert "HR_Leave_Policy" in result.cited


def test_does_not_confuse_two_documents_with_the_same_word():
    result = extract_citations(
        "Parental leave is covered by the Parental Leave policy.", CANDIDATES
    )
    assert "HR_Parental_Leave" in result.cited
    assert "HR_Leave_Policy" not in result.cited


def test_unresolved_marker_is_reported_not_guessed():
    result = extract_citations(
        "Per [Source: Finance_Moon_Base_Policy], claims are capped.", CANDIDATES
    )
    assert result.cited == []
    assert result.unknown == ["Finance_Moon_Base_Policy"]


def test_aliases_include_the_short_human_form():
    aliases = doc_aliases("HR_Leave_Policy")
    assert "Leave Policy" in aliases
    assert "HR_Leave_Policy" in aliases


def test_no_citation_is_empty_not_a_fallback_to_everything():
    result = extract_citations("I don't know.", CANDIDATES)
    assert result.cited == []
