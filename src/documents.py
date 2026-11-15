"""documents - one place that turns a file on disk into canonical IDs and chunks.

Document identity is load-bearing for every retrieval and citation metric, so
normalization happens exactly once, here, and is used by ingestion, ground truth
and the citation extractor alike.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.token_utils import encoder

logger = logging.getLogger(__name__)

_META_KEYS = {
    "document": "doc_id",
    "department": "department",
    "version": "version",
    "effective": "effective",
    "owner": "owner",
    "classification": "classification",
}


def normalize_doc_id(name: str) -> str:
    """``docs/HR_Leave_Policy.txt`` -> ``HR_Leave_Policy``.

    Canonical IDs carry no extension, no directory and no whitespace surprises,
    so ``Foo.txt`` can never fail to equal ``Foo``.
    """
    cleaned = str(name).strip().replace("\\", "/")
    stem = Path(cleaned).name
    if stem.lower().endswith(".txt"):
        stem = stem[: -len(".txt")]
    return stem.strip()


def parse_document(path: Path) -> dict:
    """Split a document into its ``key: value`` header and markdown body."""
    text = path.read_text(encoding="utf-8")
    doc_id = normalize_doc_id(path.name)
    header, separator, body = text.partition("\n---\n")
    meta: dict[str, str] = {}
    if separator:
        for line in header.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            mapped = _META_KEYS.get(key.strip().lower().replace(" ", "_"))
            if mapped:
                meta[mapped] = value.strip()
    else:
        body = text
    return {
        "doc_id": normalize_doc_id(meta.get("doc_id", doc_id)),
        "department": meta.get("department", "Unknown"),
        "version": meta.get("version", ""),
        "effective": meta.get("effective", ""),
        "owner": meta.get("owner", ""),
        "classification": meta.get("classification", ""),
        "body": body.strip(),
        "path": str(path),
    }


def _window_section(window: str) -> str:
    """Best-effort section label: the first markdown heading in the window."""
    for line in window.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or "General"
    return "General"


def chunk_document(
    doc_id: str,
    body: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Token-window a document into ``chunk_size`` tokens with ``overlap``.

    Chunk IDs are deterministic (``Doc::000``), persisted as Pinecone vector IDs
    and returned by ``query()``, so retrieval is scored on chunk identity rather
    than on the model's prose.
    """
    if not body.strip():
        return []
    tokens = encoder.encode(body)
    step = max(1, chunk_size - overlap)
    chunks: list[dict] = []
    for index, start in enumerate(range(0, len(tokens), step)):
        window_tokens = tokens[start : start + chunk_size]
        if not window_tokens:
            break
        text = encoder.decode(window_tokens).strip()
        if not text:
            continue
        chunks.append(
            {
                "chunk_id": f"{doc_id}::{index:03d}",
                "doc_id": doc_id,
                "chunk_index": index,
                "section": _window_section(text),
                "content": text,
                "tokens": len(window_tokens),
            }
        )
        if start + chunk_size >= len(tokens):
            break
    return chunks


@lru_cache(maxsize=8)
def corpus_doc_ids(directory: Path) -> tuple[str, ...]:
    """Every canonical document ID in the corpus, from filenames only.

    Citation extraction needs the full vocabulary, not just the retrieved
    subset, otherwise a fabricated citation is invisible.
    """
    return tuple(sorted(normalize_doc_id(p.name) for p in Path(directory).glob("*.txt")))


def load_documents(directory: Path) -> list[dict]:
    """Parse every ``.txt`` in a directory, sorted for deterministic chunk IDs."""
    files = sorted(directory.glob("*.txt"))
    docs = [parse_document(path) for path in files]
    logger.info("Loaded %d documents from %s", len(docs), directory)
    return docs


def corpus_stats(directory: Path) -> dict:
    """Counts and token totals, used by the ingest script's acceptance check."""
    docs = load_documents(directory)
    per_doc = []
    for doc in docs:
        chunks = chunk_document(doc["doc_id"], doc["body"])
        per_doc.append(
            {
                "doc_id": doc["doc_id"],
                "department": doc["department"],
                "tokens": len(encoder.encode(doc["body"])),
                "words": len(doc["body"].split()),
                "chunks": len(chunks),
            }
        )
    return {
        "documents": len(docs),
        "tokens": sum(d["tokens"] for d in per_doc),
        "words": sum(d["words"] for d in per_doc),
        "chunks": sum(d["chunks"] for d in per_doc),
        "per_document": per_doc,
        "by_department": {
            dept: sum(1 for d in per_doc if d["department"] == dept)
            for dept in sorted({d["department"] for d in per_doc})
        },
    }
