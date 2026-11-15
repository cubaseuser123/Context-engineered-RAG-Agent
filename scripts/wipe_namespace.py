"""Empty the Pinecone namespace so no stale vector survives the corpus rewrite.

``delete_all`` on a namespace keeps the index definition (768 dims, cosine) and
removes every vector; a stale vector sharing the namespace would quietly corrupt
retrieval recall.
"""
import argparse
import json
import logging
import sys

from src.config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from src.stores.vector_store import delete_namespace, namespace_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("wipe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Empty the Pinecone namespace.")
    parser.add_argument("--dry-run", action="store_true", help="report counts and exit without deleting")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    before = namespace_stats()
    logger.info("Before: %s", before)
    if args.dry_run:
        return 0
    after = delete_namespace()
    remaining = after.get("vector_count", 0)
    print(json.dumps({"index": PINECONE_INDEX_NAME, "namespace": PINECONE_NAMESPACE,
                      "before": before, "after": after}, indent=2))
    if remaining:
        logger.error("Namespace still reports %s vectors; Pinecone deletes are "
                     "eventually consistent - re-check in a minute", remaining)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
