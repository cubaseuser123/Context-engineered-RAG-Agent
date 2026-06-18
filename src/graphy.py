"""
graphy - LangGraph StateGraph assembly.
Wires all nodes with conditional edges based on router intent.
"""
import logging 
from typing import Literal 

from langgraph.graph import END, START, StateGraph

from src.models.state import AgentState
from src.nodes.memory_reader import memory_reader_node
from src.nodes.memory_writer import memory_writer_node
from src.nodes.out_of_scope import out_of_scope_node
from src.nodes.retriever import retriever_node
from src.nodes.router import router_node
from src.nodes.synthesizer import synthesizer_node

# We will implement context_enforcer_node next, so we use a safe import for now to prevent red lines
try:
    from src.nodes.context_enforcer import context_enforcer_node
except ImportError:
    # Dummy node until we write the real file
    def context_enforcer_node(state: AgentState) -> dict:
        return {}

logger = logging.getLogger(__name__)

def route_by_intent(state: AgentState) -> Literal["retriever", "memory_reader", "out_of_scope"]:
    """Conditional edge: route based on classified intent."""
    intent = state.get("intent", "out_of_scope")
    if intent == "policy_lookup":
        return "retriever"
    elif intent == "out_of_scope":
        return "out_of_scope"
    else:
        return "memory_reader"

def build_graph():
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("memory_reader", memory_reader_node)
    graph.add_node("context_enforcer", context_enforcer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("memory_writer", memory_writer_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "retriever": "retriever",
            "memory_reader": "memory_reader",
            "out_of_scope": "out_of_scope",
        },
    )

    graph.add_edge("retriever", "memory_reader")
    graph.add_edge("memory_reader", "context_enforcer")
    graph.add_edge("context_enforcer", "synthesizer")
    graph.add_edge("synthesizer", "memory_writer")
    graph.add_edge("memory_writer", END)

    graph.add_edge("out_of_scope", END)

    return graph

def _get_enforcer_node():
    """Import context_enforcer lazily to avoid circular deps."""
    from src.nodes.context_enforcer import context_enforcer_node
    return context_enforcer_node

def compile_agent():
    """Build, compile and return the agent."""
    graph = build_graph()
    app = graph.compile()
    logger.info("Agent graph compile successfully!")
    return app

def run_query(app, query: str, user_id: str = "default", conversation_history: list | None = None, turn: int = 0) -> dict:
    """Convenience function to invoke the agent with a query"""
    state = {
        "query" : query,
        "user_id" : user_id,
        "conversation_history" : conversation_history or [],
        "turn_number" : turn,
    }
    result = app.invoke(state)
    return result 

    