"""citations - extract which documents an answer actually cites.

The old implementation was "every retrieved document whose name appears in the
answer, otherwise every retrieved document", which measured nothing. This one
reads the ``[Source: ...]`` markers the prompt asks for plus named-document
references, resolves them against the corpus vocabulary, and reports what it
could not resolve instead of silently guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import VALID_DEPARTMENTS
from src.documents import normalize_doc_id

_MARKER_RE = re.compile(r"\[\s*source\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)

# Words too generic to identify a document on their own.
_GENERIC_ALIASES = {"policy", "policies", "standard", "guide", "process", "matrix", "commitment"}


@dataclass
class CitationExtraction:
    cited: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)


def doc_aliases(doc_id: str) -> list[str]:
    """Canonical ID plus the human forms an answer is likely to use.

    ``HR_Leave_Policy`` -> ``HR Leave Policy``, ``Leave Policy``,
    ``Leave`` is deliberately not offered: single generic tokens produce
    false positives, not citations.
    """
    canonical = normalize_doc_id(doc_id)
    words = [w for w in re.split(r"[_\s]+", canonical) if w]
    if words and words[0] in VALID_DEPARTMENTS:
        words = words[1:]

    candidates = {
        canonical,
        canonical.replace("_", " "),
        " ".join(words),
    }
    if len(words) > 2:
        candidates.add(" ".join(words[-2:]))
    aliases = []
    for candidate in candidates:
        cleaned = candidate.strip()
        if not cleaned:
            continue
        if cleaned.lower() in _GENERIC_ALIASES:
            continue
        if len(cleaned) < 8 and " " not in cleaned:
            continue
        aliases.append(cleaned)
    return sorted(set(aliases), key=len, reverse=True)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w]){re.escape(alias)}(?![\w])", re.IGNORECASE)


def extract_citations(response: str, candidates: list[str]) -> CitationExtraction:
    """Return canonical doc IDs the response cites, out of ``candidates``.

    Markers that match no candidate are returned as ``unknown`` so a fabricated
    citation shows up in citation_precision instead of disappearing.
    """
    if not response:
        return CitationExtraction()

    known: dict[str, str] = {}
    for doc_id in candidates:
        for alias in doc_aliases(doc_id):
            known.setdefault(alias.lower(), normalize_doc_id(doc_id))

    cited: list[str] = []
    unknown: list[str] = []
    markers: list[str] = []

    for raw in _MARKER_RE.findall(response):
        marker = raw.strip()
        markers.append(marker)
        resolved = known.get(normalize_doc_id(marker).lower()) or known.get(marker.lower())
        if resolved is None:
            unknown.append(normalize_doc_id(marker))
        elif resolved not in cited:
            cited.append(resolved)

    # Longest alias first, and reject any later match that overlaps a longer
    # one: in "the Parental Leave policy", "Leave Policy" must not also cite
    # HR_Leave_Policy.
    matches: list[tuple[int, int, str]] = []
    for alias, doc_id in ordered(known):
        for match in _alias_pattern(alias).finditer(response):
            matches.append((match.start(), match.end(), doc_id))
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0]))

    taken: list[tuple[int, int]] = []
    for start, end, doc_id in matches:
        if any(start < taken_end and end > taken_start for taken_start, taken_end in taken):
            continue
        taken.append((start, end))
        if doc_id not in cited:
            cited.append(doc_id)

    return CitationExtraction(cited=cited, unknown=unknown, markers=markers)


def ordered(mapping: dict[str, str]) -> list[tuple[str, str]]:
    """Longest alias first, so ``Leave Policy`` beats ``Policy``-ish fragments."""
    return sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)
