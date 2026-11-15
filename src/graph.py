"""graph - LangGraph assembly for both arms.

One set of nodes, one compiled graph per :class:`RunConfig`. The edges are what
differ:

* engineered: router -> (retriever | memory_reader | out_of_scope) -> ...
* naive:      retriever -> context_enforcer -> synthesizer -> memory_writer

The naive arm never sees the router, never reads or writes memory, and the
enforcer is a pass-through. Nothing else changes.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.models.state import AgentState
from src.nodes.context_enforcer import context_enforcer_node
from src.nodes.memory_reader import memory_reader_node
from src.nodes.memory_writer import memory_writer_node
from src.nodes.out_of_scope import out_of_scope_node
from src.nodes.retriever import retriever_node
from src.nodes.router import router_node
from src.nodes.synthesizer import synthesizer_node
from src.run_config import RunConfig

logger = logging.getLogger(__name__)


def route_by_intent(
    state: AgentState,
) -> Literal["retriever", "memory_reader", "out_of_scope"]:
    """Conditional edge: route on the classified intent."""
    intent = state.get("intent", "out_of_scope")
    if intent == "out_of_scope":
        return "out_of_scope"
    if state.get("needs_retrieval", intent in {"policy_lookup", "clarification"}):
        return "retriever"
    return "memory_reader"


def build_graph(config: RunConfig):
    """Build the graph for one arm."""
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("context_enforcer", context_enforcer_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("memory_writer", memory_writer_node)

    if config.routing:
        graph.add_node("router", router_node)
        graph.add_node("memory_reader", memory_reader_node)
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
        graph.add_edge("out_of_scope", END)
    else:
        graph.add_edge(START, "retriever")
        graph.add_edge("retriever", "context_enforcer")

    graph.add_edge("context_enforcer", "synthesizer")
    graph.add_edge("synthesizer", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph


@lru_cache(maxsize=16)
def _compile(config: RunConfig):
    app = build_graph(config).compile()
    logger.info("Compiled graph for arm=%s config=%s", config.arm, config.config_hash)
    return app


def compile_agent(config: RunConfig | None = None):
    """Compile (and cache) the agent for a given arm configuration."""
    return _compile(config or RunConfig.engineered())


def run_query(
    app,
    query: str,
    *,
    user_id: str = "default",
    conversation_history: list | None = None,
    turn: int = 0,
    config: RunConfig | None = None,
) -> dict:
    """Invoke the agent for one turn."""
    state = {
        "query": query,
        "user_id": user_id,
        "conversation_history": conversation_history or [],
        "turn_number": turn,
        "run_config": (config or RunConfig.engineered()).as_dict(),
    }
    return app.invoke(state)
