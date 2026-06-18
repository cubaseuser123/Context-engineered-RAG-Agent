"""
Constants, budget caps and model names stay here
"""
import os 
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', '')


PINECONE_INDEX_NAME : str = os.getenv('PINECONE_INDEX_NAME', 'meridian-policies')
PINECONE_CLOUD : str = 'aws'
PINECONE_REGION : str = 'us-east-1'
EMBEDDING_DIMENSION : int = 768

ROUTER_MODEL : str = 'gemini-2.5-flash'
SYNTHESIZER_MODEL : str = 'gemini-2.5-pro'
MEMORY_WRITER_MODEL : str = 'gemini-2.5-flash'
EMBEDDING_MODEL : str = 'models/text-embedding-004'

SYSTEM_PROMPT_BUDGET : int = 500
MEMORY_BUDGET : int = 500
RETRIEVAL_BUDGET : int = 3000
CONVERSATION_BUDGET : int = 1000 
TOTAL_BUDGET : int = 5000

RETRIEVAL_TOP_K : int = 8 
CHUNK_SIZE : int = 512
CHUNK_OVERLAP : int = 50 

PROJECT_ROOT : Path = Path(__file__).resolve().parent.parent
CORPUS_DIR : Path = PROJECT_ROOT / 'corpus'/ 'documents'
TEST_SUITE_PATH : Path = PROJECT_ROOT/'corpus'/'test_suite.json'
MEMORY_DB_PATH : Path = PROJECT_ROOT/'memory.db'

PHOENIX_ENDPOINT : str = 'http://127.0.0.1.6006/b1/traces'
PHOENIX_PROJECT_NAME : str = 'context-engineering-rag'

VALID_INTENTS : list[str] = ['policy_lookup', 'clarification', 'memory_recall', 'out_of_scope']
VALID_DEPARTMENTS: list[str] = ["HR", "IT", "Finance", "Legal", "Operations"]


