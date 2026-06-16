import hashlib
import logging
from pathlib import Path

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from src.config import(
    CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL,
    GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_CLOUD, PINECONE_INDEX_NAME,
    PINECONE_REGION, RETRIEVAL_TOP_K
)

logger = logging.getLogger(__name__)

_pc: Pinecone | None = None
_index = None
genai.configure(api_key=GOOGLE_API_KEY)

def _get_pc() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key = PINECONE_API_KEY)
    return _pc

def _get_index():
    global _index
    if _index is not None:
        return _index
    pc = _get_pc()
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        logger.info("Creating Pinecone index '%s'", PINECONE_INDEX_NAME)
        pc.create_index(
            name = PINECONE_INDEX_NAME,
            dimension= EMBEDDING_DIMENSION,
            metric = "cosine",
            spec= ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    _index = pc.Index(PINECONE_INDEX_NAME)
    return _index

def _embed_texts(texts: list[str]) -> list[list[float]]:
    result = genai.embed_content(model=EMBEDDING_MODEL, content=texts, task_type="retrieval_document")
    return result['embedding']

def _embed_query(text: str) -> list[float]:
    result = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_query")
    return result["embedding"]

def _chunk_id(doc: str, idx: int) -> str:
    return hashlib.md5(f"{doc}::chunk_{idx}".encode()).hexdigest()

def _parse_doc(path: Path) -> dict:
    """Parse policy document with metadata header."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 1)
    meta = {}
    if len(parts) == 2:
        for line in parts[0].strip().splitlines():
            if ":" in line:
                k,v = line.split(":", 1)
                meta[k.strip().lower().replace(" ", "_")] = v.strip()
        body = parts[1].strip()
    else:
        body = text.strip()
    return {"source_doc": meta.get("document", path.stem), "department": meta.get("department", "Unknown"), "body": body}

def _chunk_text(text: str) -> list[str]:
    """Split text into chunks."""
    char_chunk = CHUNK_SIZE * 4
    char_overlap = CHUNK_OVERLAP * 4
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start: start + char_chunk].strip()
        if chunk:
            chunks.append(chunk)
        start += char_chunk - char_overlap
    return chunks

def ingest(docs_dir: Path) -> int:
    """Ingest all .txt docs into Pinecone. Returns total chunks upserted."""
    index = _get_index()
    files = sorted(docs_dir.glob("*.txt"))
    total = 0
    for fp in files:
        doc = _parse_doc(fp)
        chunks = _chunk_text(doc["body"])
        if not chunks:
            continue
        embeddings = _embed_texts(chunks)
        vectors = [(_chunk_id(doc["source_doc"], i), emb, {"source_doc": doc["source_doc"], "department": doc["department"], "section": f"chunk_{i}", "text": ch}) for i, (ch, emb) in enumerate(zip(chunks, embeddings))]
        for b in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[b : b + 100], namespace="policies")
        total += len(chunks)
        logger.info("%s -> %d chunks", doc["source_doc"],len(chunks))
    logger.info("Ingestion done: %d chunks", total)
    return total

def query(text: str, k: int = RETRIEVAL_TOP_K, metadata_filter: str | None = None) -> list[dict]:
    """Semantic search. Returns [{content, source_doc, section, department, score}]."""
    index = _get_index()
    emb = _embed_query(text)
    filt = {"department": {"$eq": metadata_filter}} if metadata_filter else None
    results = index.query(vector=emb, top_k=k, namespace="policies", include_metadata=True, filter=filt)
    chunks = []
    for m in results.matches:
        meta = m.metadata or {}
        chunks.append({"content": meta.get("text", ""), "source_doc": meta.get("source_doc", ""), "section": meta.get("section", ""), "department": meta.get("department", ""), "score": float(m.score)})
    logger.info("Query returned %d chunks (filter=%s)", len(chunks), metadata_filter)
    return chunks