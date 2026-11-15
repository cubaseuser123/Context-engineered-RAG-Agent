import json

import src.llm as llm


def _payload(text, prompt=7, completion=3):
    return {
        "id": "req_1",
        "model": "qwen/qwen3.8-27b",
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_key_cursor", {})
    monkeypatch.setattr(llm, "_key_cooldowns", {})
    monkeypatch.setattr(llm, "GROQ_API_KEYS", ["test-key"])
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "QUOTA_LEDGER_PATH", tmp_path / "quota.json")
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


def test_chat_returns_provider_usage_and_strips_json_fences(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    seen = []

    def fake_post(url, payload, headers):
        seen.append(payload)
        return _payload('```json\n{"a": 1}\n```'), {"x-ratelimit-remaining-requests": "9"}

    monkeypatch.setattr(llm, "_http_post_json", fake_post)
    response = llm.chat(
        [{"role": "user", "content": "hi"}], model="qwen/qwen3.8-27b", json_mode=True
    )

    assert response.text == '{"a": 1}'
    assert json.loads(response.text) == {"a": 1}
    assert response.usage.total_tokens == 10
    assert response.cached is False
    assert response.rate_limits["x-ratelimit-remaining-requests"] == 9
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_second_identical_call_hits_the_cache(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    calls = []

    def fake_post(url, payload, headers):
        calls.append(payload)
        return _payload("answer"), {}

    monkeypatch.setattr(llm, "_http_post_json", fake_post)
    first = llm.chat([{"role": "user", "content": "hi"}], model="qwen/qwen3.8-27b")
    second = llm.chat([{"role": "user", "content": "hi"}], model="qwen/qwen3.8-27b")

    assert len(calls) == 1
    assert first.cached is False and second.cached is True
    assert second.text == "answer"


def test_reasoning_args_are_dropped_when_the_model_rejects_them(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "GROQ_REASONING_ARGS", {"reasoning_effort": "low"})
    attempts = []

    def fake_post(url, payload, headers):
        attempts.append(dict(payload))
        if len(attempts) == 1:
            raise llm.ProviderError("HTTP 400: unknown parameter reasoning_effort")
        return _payload("ok"), {}

    monkeypatch.setattr(llm, "_http_post_json", fake_post)
    response = llm.chat([{"role": "user", "content": "hi"}], model="qwen/qwen3.8-27b")

    assert response.text == "ok"
    assert "reasoning_effort" in attempts[0]
    assert "reasoning_effort" not in attempts[1]


def test_missing_key_is_reported_not_retried(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "GROQ_API_KEYS", [])
    try:
        llm.chat([{"role": "user", "content": "hi"}], model="m")
    except llm.ProviderNotConfigured as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("expected ProviderNotConfigured")


def test_successive_requests_use_different_keys(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "GROQ_API_KEYS", ["key-a", "key-b", "key-c", "key-d"])
    monkeypatch.setattr(llm, "_key_cursor", {}, raising=False)
    seen = []
    def post(url, payload, headers):
        seen.append(headers["Authorization"])
        return _payload("ok"), {}
    monkeypatch.setattr(llm, "_http_post_json", post)
    for i in range(4):
        llm.chat([{"role": "user", "content": str(i)}], model="m", use_cache=False)
    assert seen == ["Bearer key-a", "Bearer key-b", "Bearer key-c", "Bearer key-d"]


def test_quota_message_containing_400_does_not_drop_reasoning_args(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "GROQ_REASONING_ARGS", {"reasoning_effort": "low"})
    seen = []
    def post(url, payload, headers):
        seen.append(dict(payload))
        if len(seen) == 1:
            raise llm.ProviderError("HTTP 429: token quota Used 140000, Requested 4000")
        return _payload("ok"), {}
    monkeypatch.setattr(llm, "_http_post_json", post)
    llm.chat([{"role": "user", "content": "test"}], model="m", use_cache=False)
    assert seen[1]["reasoning_effort"] == "low"


def test_rate_limit_tries_another_key_before_sleeping(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "GROQ_API_KEYS", ["a", "b"])
    monkeypatch.setattr(llm, "_key_cursor", {})
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    seen = []
    def post(url, payload, headers):
        seen.append(headers["Authorization"])
        if len(seen) == 1:
            raise llm.ProviderError("HTTP 429: quota")
        return _payload("ok"), {}
    monkeypatch.setattr(llm, "_http_post_json", post)
    llm.chat([{"role": "user", "content": "test"}], model="m", use_cache=False)
    assert seen == ["Bearer a", "Bearer b"]
    assert sleeps == []


def test_daily_limited_key_is_skipped_on_later_requests(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "GROQ_API_KEYS", ["a", "b"])
    monkeypatch.setattr(llm, "_key_cursor", {})
    monkeypatch.setattr(llm, "_key_cooldowns", {}, raising=False)
    seen = []
    def post(url, payload, headers):
        seen.append(headers["Authorization"])
        if seen[-1] == "Bearer a":
            raise llm.ProviderError("HTTP 429: daily quota", headers={"retry-after": "1200"})
        return _payload("ok"), {}
    monkeypatch.setattr(llm, "_http_post_json", post)
    for _ in range(2):
        llm.chat([{"role": "user", "content": "test"}], model="m", use_cache=False)
    assert seen == ["Bearer a", "Bearer b", "Bearer b"]


def test_long_pool_cooldown_surfaces_without_sleeping_for_minutes(monkeypatch, tmp_path):
    import pytest
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(llm, "_key_cooldowns", {}, raising=False)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    def post(*args):
        raise llm.ProviderError("HTTP 429: daily quota", headers={"retry-after": "1200"})
    monkeypatch.setattr(llm, "_http_post_json", post)
    with pytest.raises(llm.ProviderRateLimited, match="cooling down"):
        llm.chat([{"role": "user", "content": "test"}], model="m", use_cache=False)
    assert sleeps == []
