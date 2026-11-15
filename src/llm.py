"""llm - the one place that talks to a model provider.

Design (docs/superpowers/specs/2026-11-03-observability-evals-design.md §8.2.2):

* ``chat()``      -> Groq's OpenAI-compatible endpoint, returns text + usage.
* ``embed()``     -> Google AI Studio ``gemini-embedding-001`` at 768 dims.

Groq has no embedding endpoint, so the study necessarily spans two providers.
Everything that three copies of provider boilerplate used to do - backoff, key
rotation, response caching, the quota ledger, rate-limit capture - lives here.

Both functions return provider-reported usage, never a tiktoken estimate.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from langfuse import observe

from src.config import (
    CACHE_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEYS,
    GROQ_BASE_URL,
    GROQ_REASONING_ARGS,
    QUOTA_LEDGER_PATH,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 6
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 45.0
REQUEST_TIMEOUT_SECONDS = 120
_key_cursor: dict[str, int] = {}
_key_cooldowns: dict[tuple[str, int], float] = {}


class ProviderError(RuntimeError):
    """Any provider failure that is not retried away."""

    def __init__(self, message: str, *, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}


class ProviderNotConfigured(ProviderError):
    """A required key is missing; the caller should surface this, not retry."""


class ProviderRateLimited(ProviderError):
    """Retries and key rotation were both exhausted."""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_api(cls, usage: dict[str, Any] | None) -> "Usage":
        usage = usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    usage: Usage
    latency_ms: float
    cached: bool = False
    request_id: str | None = None
    rate_limits: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage.as_dict(),
            "latency_ms": round(self.latency_ms, 1),
            "cached": self.cached,
            "request_id": self.request_id,
            "rate_limits": self.rate_limits,
        }


# ---------------------------------------------------------------------------
# response cache - keyed by everything that can change the answer
# ---------------------------------------------------------------------------
def _cache_key(kind: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def _cache_read(key: str) -> dict[str, Any] | None:
    if os.getenv("DISABLE_LLM_CACHE", "0") == "1":
        return None
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable cache entry %s", path)
        return None


def _cache_write(key: str, value: dict[str, Any]) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# quota ledger - runs/quota.json, one bucket per provider:model per day
# ---------------------------------------------------------------------------
def _ledger_add(
    provider: str,
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    calls: int = 0,
    cached: int = 0,
    errors: int = 0,
) -> None:
    path = QUOTA_LEDGER_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        ledger = {}

    day = date.today().isoformat()
    bucket_name = f"{provider}:{model}"
    bucket = ledger.setdefault(day, {}).setdefault(
        bucket_name,
        {
            "calls": 0,
            "cached_calls": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
    )
    bucket["calls"] += calls
    bucket["cached_calls"] += cached
    bucket["errors"] += errors
    bucket["prompt_tokens"] += prompt_tokens
    bucket["completion_tokens"] += completion_tokens

    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_ledger() -> dict[str, Any]:
    """Daily spend per provider:model, for the run manifest."""
    path = QUOTA_LEDGER_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------
_RATE_HEADER_PREFIXES = ("x-ratelimit-", "retry-after")


def _capture_rate_limits(headers: dict[str, str]) -> dict[str, int]:
    captured: dict[str, int] = {}
    for name, value in headers.items():
        if not name.startswith(_RATE_HEADER_PREFIXES):
            continue
        try:
            captured[name] = int(float(value))
        except (TypeError, ValueError):
            continue
    return captured


def _http_post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "context-engineering/1.0", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
            raw_headers = {k.lower(): v for k, v in response.headers.items()}
            return body, raw_headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise ProviderError(
            f"HTTP {exc.code}: {detail}", headers=dict(exc.headers or {})
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"transport failure: {exc.reason}") from exc


def _backoff_seconds(attempt: int, headers: dict[str, str] | None = None) -> float:
    if headers and "retry-after" in headers:
        try:
            return min(float(headers["retry-after"]), MAX_BACKOFF_SECONDS)
        except ValueError:
            pass
    delay = min(BASE_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
    return delay + random.uniform(0, 0.75)


def _is_rate_limit(message: str) -> bool:
    lowered = message.lower()
    return "429" in message or "rate limit" in lowered or "quota" in lowered


def _strip_code_fences(text: str) -> str:
    """JSON mode should already be clean; a reasoning model still sometimes isn't."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    first = min(
        (idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0),
        default=-1,
    )
    if first > 0:
        stripped = stripped[first:]
    last = max(stripped.rfind("}"), stripped.rfind("]"))
    if last >= 0 and last + 1 < len(stripped):
        stripped = stripped[: last + 1]
    return stripped.strip()


@observe(as_type="generation", name="llm.chat")
def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    json_mode: bool = False,
    max_tokens: int | None = None,
    use_cache: bool = True,
) -> LLMResponse:
    """One chat completion against Groq. Returns text and provider usage."""
    if not messages:
        raise ValueError("chat() needs at least one message")
    if not GROQ_API_KEYS:
        raise ProviderNotConfigured(
            "No Groq key. Set GROQ_API_KEY (or GROQ_API_KEYS for a pool) in .env."
        )

    key = _cache_key(
        "chat",
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
        },
    )
    if use_cache:
        hit = _cache_read(key)
        if hit is not None:
            _ledger_add("groq", model, cached=1, calls=0)
            return LLMResponse(
                text=hit["text"],
                model=model,
                usage=Usage(),
                latency_ms=0.0,
                cached=True,
                request_id=hit.get("request_id"),
            )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload.update(GROQ_REASONING_ARGS)

    last_error = "unknown provider failure"
    for attempt in range(MAX_ATTEMPTS):
        start_key = _key_cursor.get(model, 0) % len(GROQ_API_KEYS)
        order = [(start_key + offset) % len(GROQ_API_KEYS)
                 for offset in range(len(GROQ_API_KEYS))]
        now = time.monotonic()
        key_index = next((index for index in order
                          if _key_cooldowns.get((model, index), 0) <= now), None)
        if key_index is None:
            key_index = min(order, key=lambda index: _key_cooldowns[(model, index)])
            wait = _key_cooldowns[(model, key_index)] - now
            if wait > MAX_BACKOFF_SECONDS:
                raise ProviderRateLimited(
                    f"Groq {model}: all keys cooling down; retry after {wait:.0f} seconds"
                )
            logger.info("Groq keys cooling down; waiting %.1f seconds", wait)
            time.sleep(wait)
            _key_cooldowns.pop((model, key_index), None)
        api_key = GROQ_API_KEYS[key_index]
        _key_cursor[model] = (key_index + 1) % len(GROQ_API_KEYS)
        started = time.perf_counter()
        try:
            body, headers = _http_post_json(
                f"{GROQ_BASE_URL}/chat/completions",
                payload,
                {"Authorization": f"Bearer {api_key}"},
            )
        except ProviderError as exc:
            message = str(exc)
            last_error = message
            _ledger_add("groq", model, calls=1, errors=1)
            rejected_args = [arg for arg in GROQ_REASONING_ARGS if arg in message]
            if message.startswith("HTTP 400:") and rejected_args:
                # Some model IDs reject reasoning_effort/reasoning_format. That
                # is a parameter mismatch, not a fatal error: drop them once.
                for arg in rejected_args:
                    payload.pop(arg, None)
                logger.info("Groq rejected reasoning args; retrying without them")
                continue
            if not _is_rate_limit(message) and "5" not in message[:8]:
                raise
            if _is_rate_limit(message):
                try:
                    wait = max(0.0, float(exc.headers["retry-after"]))
                except (KeyError, ValueError):
                    wait = _backoff_seconds(attempt)
                _key_cooldowns[(model, key_index)] = time.monotonic() + wait
                continue  # try another available key before waiting
            time.sleep(_backoff_seconds(attempt))
            continue

        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Malformed Groq response: {body}") from exc

        usage = Usage.from_api(body.get("usage"))
        from src.tracing import get_client
        tracing_client = get_client()
        if tracing_client:
            tracing_client.update_current_generation(
                model=body.get("model", model), output=text,
                usage_details={"input": usage.prompt_tokens, "output": usage.completion_tokens,
                               "total": usage.total_tokens},
            )
        if json_mode:
            text = _strip_code_fences(text)

        result = LLMResponse(
            text=text.strip(),
            model=body.get("model", model),
            usage=usage,
            latency_ms=latency_ms,
            cached=False,
            request_id=body.get("id") or headers.get("x-request-id"),
            rate_limits=_capture_rate_limits(headers),
        )
        if use_cache:
            _cache_write(
                key,
                {
                    "text": result.text,
                    "request_id": result.request_id,
                    "usage": usage.as_dict(),
                },
            )
        _ledger_add(
            "groq",
            model,
            calls=1,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return result

    raise ProviderRateLimited(
        f"Groq {model} failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# embeddings - Google AI Studio
# ---------------------------------------------------------------------------
_google_client: Any = None


def _get_google_client() -> Any:
    global _google_client
    if _google_client is not None:
        return _google_client
    if not GOOGLE_API_KEY:
        raise ProviderNotConfigured(
            "No Google AI Studio key. Set GOOGLE_API_KEY in .env (embeddings only)."
        )
    try:
        from google import genai  # imported lazily: only embeddings need it
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ProviderNotConfigured(
            "google-genai is not installed; run `pip install google-genai`."
        ) from exc
    _google_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _google_client


def _embed_batch(
    batch: list[str], task_type: str, model: str
) -> tuple[list[list[float]], dict[str, int]]:
    from google.genai import types

    client = _get_google_client()
    last_error = "unknown embedding failure"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIMENSION,
                ),
            )
            vectors = [list(item.values) for item in response.embeddings]
            if len(vectors) != len(batch):
                raise ProviderError(
                    f"Embedding count mismatch: {len(vectors)} for {len(batch)} inputs"
                )
            return vectors, {}
        except Exception as exc:  # provider SDK raises a wide variety
            last_error = str(exc)
            if not _is_rate_limit(last_error) and "5" not in last_error[:8]:
                raise ProviderError(f"Embedding failed: {last_error}") from exc
            logger.warning(
                "Embedding request %d/%d rate limited, backing off", attempt + 1, MAX_ATTEMPTS
            )
            time.sleep(_backoff_seconds(attempt))
    raise ProviderRateLimited(f"Embedding failed after {MAX_ATTEMPTS} attempts: {last_error}")


def embed(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = EMBEDDING_MODEL,
    use_cache: bool = True,
) -> list[list[float]]:
    """Embed texts with Gemini at 768 dims, batching and caching per string."""
    if not texts:
        return []

    vectors: list[list[float] | None] = [None] * len(texts)
    pending: list[int] = []
    for idx, text in enumerate(texts):
        key = _cache_key(
            "embed",
            {
                "model": model,
                "task_type": task_type,
                "dims": EMBEDDING_DIMENSION,
                "text": text,
            },
        )
        hit = _cache_read(key) if use_cache else None
        if hit is not None:
            vectors[idx] = hit["vector"]
            _ledger_add("google", model, cached=1)
        else:
            pending.append(idx)

    for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
        batch_idx = pending[start : start + EMBEDDING_BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]
        batch_vectors, _ = _embed_batch(batch_texts, task_type, model)
        for i, vector in zip(batch_idx, batch_vectors):
            if len(vector) != EMBEDDING_DIMENSION:
                raise ProviderError(
                    f"Expected {EMBEDDING_DIMENSION} dims, got {len(vector)}"
                )
            vectors[i] = vector
            if use_cache:
                key = _cache_key(
                    "embed",
                    {
                        "model": model,
                        "task_type": task_type,
                        "dims": EMBEDDING_DIMENSION,
                        "text": texts[i],
                    },
                )
                _cache_write(key, {"vector": vector})
        _ledger_add("google", model, calls=1)

    return [v for v in vectors if v is not None]


def embed_query(text: str, *, model: str = EMBEDDING_MODEL) -> list[float]:
    return embed([text], task_type="RETRIEVAL_QUERY", model=model)[0]


def embedding_probe(requests: int = 21) -> dict[str, Any]:
    """P2 gate: does the Google key meter embeddings separately from generation?

    Sends single-text requests so a 429 pinpoints the exact request that broke.
    Result is recorded in the run manifest rather than guessed at.
    """
    probe: dict[str, Any] = {"attempted": requests, "ok": True, "failed_at": None}
    for i in range(requests):
        try:
            embed([f"meridian probe {i}"], task_type="RETRIEVAL_DOCUMENT", use_cache=False)
        except ProviderError as exc:
            probe.update(ok=False, failed_at=i, error=str(exc))
            logger.warning("Embedding probe failed on request %d: %s", i + 1, exc)
            break
    probe["ledger"] = read_ledger().get(date.today().isoformat(), {})
    return probe


def configured() -> dict[str, Any]:
    """What the adapter can see, for preflight messages."""
    return {
        "groq_keys": len(GROQ_API_KEYS),
        "google_key": bool(GOOGLE_API_KEY),
        "cache_dir": str(CACHE_DIR),
        "ledger": str(QUOTA_LEDGER_PATH),
    }
