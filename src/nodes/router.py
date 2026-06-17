"""
router - Intent classification node. 
Takes: query + conversation_history. Returns: intent, reasoning, optional department filter. 

"""
import json
import logging

import google.generativeai as genai 

from src.config import GOOGLE_API_KEY, ROUTER_MODEL, VALID_DEPARTMENTS, VALID_INTENTS
from src.models.state import AgentState

logger = logging.getLogger(__name__)

genai.configure(api_key = GOOGLE_API_KEY)

ROUTER_SYSTEM_PROMPT = """You are a query intent classifier for a company policy assistant at Meridian Technologies. 

Classify the user's query into exactly one intent:
- "policy_lookup": User wants information from a company policy document.
- "clarification": User is following up or asking for clarification on something already discussed.
- "memory_recall": User is asking about something they previously told you (their role, preferences, etc).
- "out_of_scope": Query has nothing to do with company policy.

If the intent is "policy_lookup", also infer which department the query relates to if obvious:
HR, IT, Finance, Legal, Operations. Set department_filter to null if unclear.

Respond with ONLY valid JSON:
{"intent": "...", "reasoning": "...", "department_filter": "..." or null}

Examples:
User: "What is the travel reimbursement limit?"
{"intent": "policy_lookup", "reasoning": "User asks about travel reimbursement which falls under Finance policies", "department_filter": "Finance"}

User: "Can you explain that last point in more detail?"
{"intent": "clarification", "reasoning": "User references previous conversation with 'that last point'", "department_filter": null}

User: "What's the weather like today?"
{"intent": "out_of_scope", "reasoning": "Weather has nothing to do with company policy", "department_filter": null}
"""

def router_node(state : AgentState) -> dict:
    """Classify user query intent. Returns intent, reasoning and optional department filter."""
    query = state.get("query", "")
    history = state.get("conversation_history", [])

    #only pass last 3 turns
    recent_history = history[-3:] if len(history) > 3 else history 
    history_text = ""
    for turn in recent_history:
        history_text += f"{turn.get('role', 'user')} : {turn.get('content', '')}\n"
    user_message = f"Conversation history:\n{history_text}\n\nCurrent query: {query}"

    model = genai.GenerativeModel(
        ROUTER_MODEL,
        system_instruction=ROUTER_SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    response = model.generate_content(user_message)
    raw = response.text.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Router returned invalid JSON: %s", raw)
        parsed = {"intent" : "out_of_scope", "reasoning" : "Failed to parse router output", "department_filter" : None}

    intent = parsed.get("intent", "out_of_scope")
    if intent not in VALID_INTENTS:
        logger.warning("Router returned invalid '%s', defaulting to out_of_scope", intent)
        intent = "out_of_scope"
    
    dept = parsed.get("department_filter")
    if dept and dept not in VALID_DEPARTMENTS:
        dept = None
    
    logger.info("Router: intent=%s, dept=%s, reasoning=%s", intent, dept, parsed.get("reasoning", ""))

    return{
        "intent" : intent,
        "router_reasoning" : parsed.get("reasoning", ""),
        "department_filter" : dept,
    }
