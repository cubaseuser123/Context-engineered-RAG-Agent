"""
ingest documents - CLI script to ingest policy documents into Pinecone.
"""
import logging 
import sys 

from src.config import CORPUS_DIR
from src.stores.vector_store import ingest 

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(name)s] [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    if not CORPUS_DIR.exists():
        logger.error("Generate the corpus first, then run the script.")
        sys.exit(1)

    text_files = []
    for file_path in CORPUS_DIR.glob("*.txt"):
        text_files.append(file_path)

    doc_count = len(text_files)

    if doc_count == 0:
        logger.error("No .txt files found in %s", CORPUS_DIR)
        sys.exit(1)
        
    logger.info("Starting ingestion of %d documents from %s", doc_count, CORPUS_DIR)
    total_chunks = ingest(CORPUS_DIR)
    logger.info("Done. %d total chunks upserted to Pinecone.", total_chunks)

if __name__ == "__main__":
    main()