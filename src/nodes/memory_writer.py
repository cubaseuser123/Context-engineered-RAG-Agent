"""
memory_writer - Extract and persist user facts after synthesis.
Takes: query, response, user_id. Returns:
new_memory_entries.
"""
import json 
import logging

import google.generativeai as genai

from src.config import GOOGLE_API_KEY, MEMORY_WRITER_MODEL
from src.models.state import AgentState
from src.stores import memory_store

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

EXTRACTION_PROMPT = """Analyze this conversation exchange and extract any user-specific facts worth remembering for future conversations.

Facts to extract:
- User's role or department
- User's preferences or constraints
- Specific topics the user is tracking
- Open questions the user mentioned

Return a JSON array of strings. Each string is one fact.
Return an empty array [] if nothing worth remembering.

User query: {query}
Assistant response: {response}

JSON array:"""

def memory_writer_node(state: AgentState) -> dict:
    """Extract and store new user facts from this exchange."""
    intent = state.get("intent", "")
    if intent == "out_of_scope":
        logger.info("Memory writer: skipped (out_of_scope)")
        return {"new_memory_entries": []}

    query = state.get("query", "")
    response = state.get("response", "")
    user_id = state.get("user_id", "default")
    turn = state.get("turn_number", 0)

    model = genai.GenerativeModel(
        MEMORY_WRITER_MODEL,
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    prompt = EXTRACTION_PROMPT.format(query = query, response = response)
    result = model.generate_content(prompt)

    try:
        facts = json.loads(result.text.strip())
        if not isinstance(facts, list):
            facts = []
        facts = [f for f in facts if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        logger.warning("Memory writer: JSON parse error, %s", result.text)
        facts = []
        
    if facts:
        memory_store.write(user_id, facts, turn)
        memory_store.prune(user_id)

    logger.info("Memory writer: %d facts extracted and stored", len(facts))

    return {"new_memory_entries": [{"fact": f} for f in facts]}