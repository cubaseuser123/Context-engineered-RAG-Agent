"""refusal - deterministic detector for "the agent declined".

The old detector was a four-phrase substring match on one question and scored a
correct refusal as a hallucination. This one recognises the two refusal paths
the agent actually has (the out-of-scope reply and the empty-context reply) and
nothing else, so the confusion matrix means what it says.
"""
from __future__ import annotations

import re

from src.prompts import EMPTY_CONTEXT_RESPONSE, REFUSAL_RESPONSE

_MARKERS = (
    EMPTY_CONTEXT_RESPONSE,
    REFUSAL_RESPONSE,
    "i don't have that information in the available policy documents",
    "i do not have that information in the available policy documents",
)

_REFUSAL_RE = re.compile(
    r"\b(i (?:don't|do not) have (?:that|this|the) information|"
    r"i (?:can't|cannot) (?:help|answer)|"
    r"not (?:covered|available) in the (?:available )?policy documents|"
    r"could you rephrase your question)\b",
    re.IGNORECASE,
)


def is_refusal(response: str) -> bool:
    text = (response or "").strip()
    if not text:
        return False
    lowered = text.lower()
    for marker in _MARKERS:
        if marker.lower() in lowered:
            return True
    return bool(_REFUSAL_RE.search(text))
