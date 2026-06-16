import logging 
import sqlite3
from datetime import datetime, timezone
from src.config import MEMORY_DB_PATH

logger = logging.getLogger(__name__)

_DDL = """
Create TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    fact TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source_turn INTEGER NOT NULL
    );
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
"""

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    return conn 

def read(user_id : str) -> list[dict]: 
    conn = _get_conn()
    try: 
        rows = conn.execute(
            "SELECT fact, timestamp, source_turn FROM memories WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
        ).fetchall()
        logger.info("Memory read user = '%s': %d entries", user_id, len(rows))
        return [dict(r) for r in rows]
    finally:
        conn.close()

def write(user_id: str, facts:list[str], turn: int) -> int: 
    """Batch insert new facts. Returns count written."""
    if not facts:
        return 0
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()
    try: 
        conn.executemany(
            "INSERT INTO memories (user_id, fact, timestamp, source_turn) VALUES (?, ?, ?, ?)",
            [(user_id, f, now, turn) for f in facts],
        )
        conn.commit()
        logger.info("Memory write user='%s': %d entries", user_id, len(facts))
        return len(facts)
    finally:
        conn.close()

def prune(user_id: str, max_entries: int = 20) -> int:
    """Delete oldest entries beyond cap. Return count deleted."""
    conn = _get_conn()
    try:
        count = conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)).fetchone()[0]
        if count <= max_entries:
            return 0
        to_del = count - max_entries
        conn.execute(
            "DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE user_id =? ORDER BY timestamp ASC LIMIT ?)",
            (user_id, to_del),
        )
        conn.commit()
        logger.info("Memory prune user='%s': %d deleted", user_id, to_del)
        return to_del
    finally:
        conn.close()

def clear(user_id: str) -> int:
    """Delete all entries for a user. Testing utility."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
        


 