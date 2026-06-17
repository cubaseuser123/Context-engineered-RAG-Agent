"""
out_of_scope - Fixed refusal handler for out-of-scope queries.
"""
import logging 
from src.models.state import AgentState

logger = logging.getLogger(__name__)

REFUSAL_RESPONSE = (
    "I'm a policy assistant for Meridian Technologies. I can help you with questions about "
    "company policies including HR, IT & Security, Finance, Legal & Compliance, and Operations. "
    "Could you rephrase your question to relate to a company policy?"
)

def out_of_scope_node(state: AgentState) -> dict:
    """Return a fixed polite refusal without any LLM call."""
    logger.info("Out of scope handler: returning fixed refusal")
    return {
        "response" : REFUSAL_RESPONSE,
        "cited_sources" : [],
    }