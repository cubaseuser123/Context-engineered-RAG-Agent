"""pricing - a pinned price table, labelled approximate.

The study runs on free tiers, so realised spend is $0 and this table exists only
to keep the cost axis comparable if a paid key were ever used. Prices are list
prices remembered, not fetched: the report always says "approximate".
"""
from __future__ import annotations

LABEL = "estimated from a pinned price table, excluding router, memory and judge calls; it is not a billing measurement"

# USD per million tokens: (input, output)
PRICES: dict[str, tuple[float, float]] = {
    "qwen/qwen3.8-27b": (0.29, 0.59),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Cost of one call, or ``None`` when the model has no pinned price."""
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return (prompt_tokens / 1_000_000) * input_price + (
        completion_tokens / 1_000_000
    ) * output_price
