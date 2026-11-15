import pytest
from evals.judges import _parse_verdict


def test_citation_prompt_uses_full_policy_and_keeps_conversation_without_unrelated_documents(monkeypatch):
    from evals import judges
    from src.models.run_record import RunRecord
    passage = 'A perpetual software licence must be capitalised. ' * 8
    monkeypatch.setattr(judges, '_policy_documents', lambda: {'Finance': passage})
    record = RunRecord(run_id='r', record_id='s:00', scenario_id='s', turn_index=0,
                       arm='naive', prompt='=== MEMORY ===\nI work in IT\n\n=== DOCUMENTS ===\n[Source: Unrelated]\nUnrelated onboarding material\n\n=== CONVERSATION ===\nuser: My licence type is unknown.\n\n==== CURRENT ====\nSummarise it')
    record.output.citations = ['Finance']
    prompt = judges._prompt('citation_support', record, None)
    assert prompt.count(passage) == 1
    assert 'user: My licence type is unknown.' in prompt
    assert 'I work in IT' in prompt
    assert 'Unrelated onboarding material' not in prompt
    assert 'Unrelated onboarding material' in judges._prompt('groundedness', record, None)


def test_only_citation_grading_gets_a_new_version():
    from evals.judges import judge_version, JUDGE_VERSION
    assert judge_version('citation_support') != JUDGE_VERSION
    assert judge_version('answer_correctness') == JUDGE_VERSION
    assert judge_version('groundedness') == JUDGE_VERSION


@pytest.mark.parametrize("score", [2, -1, 0.5, True, "1"])
def test_invalid_binary_scores_are_not_silently_accepted(score):
    import json
    value, _, error = _parse_verdict(json.dumps({"score": score, "reason": "test"}))
    assert value is None
    assert error


def test_score_cannot_contradict_its_checks():
    value, _, error = _parse_verdict('{"score":1,"checks":{"answers_question":true,"claims_supported":false,"conditions_preserved":true}}')
    assert value is None
    assert error == "score contradicts correctness checks"


def test_completed_judgments_are_checkpointed_before_later_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from evals import judges
    result = judges.JudgeResult("s:00", "r", "answer_correctness", 1, "ok", "v", "m", "now")
    calls = []
    def grade(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("provider quota")
        return result
    monkeypatch.setattr(judges, "judge_record", grade)
    dataset = SimpleNamespace(by_id=lambda _: SimpleNamespace(expected_for=lambda _: object()))
    record = SimpleNamespace(error=None, scenario_id="s", turn_index=0)
    path = tmp_path / "judgements.jsonl"
    with pytest.raises(RuntimeError, match="provider quota"):
        judges.judge_records([record], dataset, metrics=("answer_correctness", "groundedness"),
                             on_result=lambda j: judges.append_judgement(path, j))
    assert judges.read_judgements(path) == [result]
