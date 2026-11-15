"""Constants, model IDs, budget caps and repo paths.

Per docs/superpowers/specs/2026-11-03-observability-evals-design.md:

* Groq hosts generation (``qwen/qwen3.8-27b``) and judging
  (``openai/gpt-oss-120b``).
* Google AI Studio supplies embeddings (``gemini-embedding-001``, 768 dims);
  Groq has no embedding endpoint, so this split is not optional.
* No Vertex AI SDK is used anywhere in ``src/``.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _split_keys(raw: str) -> list[str]:
    """Accept ``k1,k2`` (or semicolons) so several free-tier keys can pool."""
    return [k.strip() for k in raw.replace(";", ",").split(",") if k.strip()]


GROQ_API_KEYS: list[str] = _split_keys(
    os.getenv("GROQ_API_KEYS", "") or os.getenv("GROQ_API_KEY", "")
)
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")

PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "meridian-policies")
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"
PINECONE_NAMESPACE: str = "policies"
EMBEDDING_DIMENSION: int = 768
EMBEDDING_MODEL: str = "gemini-embedding-001"
# The embeddings free tier is small; batches of 100 keep one request cheap.
EMBEDDING_BATCH_SIZE: int = 100

# One generator for every arm. Swapping it mid-study is a scope violation.
GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "qwen/qwen3.8-27b")
ROUTER_MODEL: str = GENERATOR_MODEL
SYNTHESIZER_MODEL: str = GENERATOR_MODEL
MEMORY_WRITER_MODEL: str = GENERATOR_MODEL

JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
CROSS_JUDGE_MODEL: str = os.getenv("CROSS_JUDGE_MODEL", "openai/gpt-oss-20b")

# Evaluation runs are temperature 0 in both arms, by design.
EVAL_TEMPERATURE: float = 0.0
JUDGE_TEMPERATURE: float = 0.0

# Qwen 3.8 is a reasoning model: reasoning tokens are billed as completion and
# would leak prose into JSON. These are sent to Groq, and dropped on a 400.
GROQ_REASONING_ARGS: dict = {"reasoning_format": "hidden", "reasoning_effort": "low"}

SYSTEM_PROMPT_BUDGET: int = 500
MEMORY_BUDGET: int = 500
RETRIEVAL_BUDGET: int = 3000
CONVERSATION_BUDGET: int = 1000
TOTAL_BUDGET: int = 5000

RETRIEVAL_TOP_K: int = 8
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = PROJECT_ROOT / "corpus" / "documents"
MEMORY_DB_PATH: Path = PROJECT_ROOT / "memory.db"

EVALS_DIR: Path = PROJECT_ROOT / "evals"
DATASETS_DIR: Path = EVALS_DIR / "datasets"
DATASET_VERSION: str = "v1"
SCENARIOS_PATH: Path = DATASETS_DIR / "scenarios.v1.json"
CALIBRATION_LABELS_PATH: Path = DATASETS_DIR / "calibration.v1.labels.jsonl"
BASELINE_PATH: Path = PROJECT_ROOT / "evals" / "baselines" / "scorecard.v1.json"

RUNS_DIR: Path = PROJECT_ROOT / "runs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "llm"
QUOTA_LEDGER_PATH: Path = RUNS_DIR / "quota.json"

LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL: str = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

VALID_INTENTS: list[str] = [
    "policy_lookup",
    "clarification",
    "memory_recall",
    "out_of_scope",
]
VALID_DEPARTMENTS: list[str] = ["HR", "IT", "Finance", "Legal", "Operations"]
