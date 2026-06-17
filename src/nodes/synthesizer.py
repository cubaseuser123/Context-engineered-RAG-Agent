"""
synthesizer - Main generation node using Gemini Pro.
Takes : trimmed context from all 4 zones. Returns : response + cited_sources. 
This is the only expensive LLM call in the graph.
"""

import logging 

import google.generativeai as genai

from src.config import GOOGLE_API_KEY, SYNTHESIZER_MODEL
from src.models.state import AgentState

logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are a policy assistant for Meridian Technologies, a mid-sized B2B SaaS company (~800 employees, HQ Pune).

RULES:
1. Answer ONLY based on the policy documents provided in the context below.
2. ALWAYS cite the source document name in your answer (e.g. "According to the Leave Policy...").
3. If the answer is not in the provided documents, say "I don't have that information in the available policy documents."
4. NEVER invent policy details, phone numbers, email addresses, or any information not explicitly in the documents.
5. If documents contain conflicting information, acknowledge the conflict and present both versions.
6. Be concise and direct. Employees want quick answers.

You will receive context in these sections:
- MEMORY: Facts about this user from previous conversations
- DOCUMENTS: Relevant policy document excerpts
- CONVERSATION: Recent exchange history"""

def _build_context(state: AgentState) -> str:
    """Assemble the 4 context zones into a single user message."""
    parts = []
    memory = state.get("trimmed_memory", [])
    if memory:
        mem_text = "\n".join(f"- {m.get('fact', '')}" for m in memory)
        parts.append(f"=== Memory ===\n{mem_text}")


    chunks = state.get("trimmed_chunks", [])
    if chunks:
        doc_text = ""
        for c in chunks:
            doc_text += f"\n[Source: {c.get('source_doc', 'unkown')}]\n{c.get('content', '')}\n"
        parts.append(f"=== DOCUMENTS ==={doc_text}")

    history = state.get("trimmed_history", [])
    if history: 
        hist_text = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history)
        parts.append(f"=== CONVERSATION ===\n{hist_text}")

    parts.append(f"==== CURRENT ====\n{state.get('query', '')}")
    return "\n\n".join(parts)

def synthesizer_node(state: AgentState) -> dict:
    """Generate the final response with source citations."""
    context = _build_context(state)
    model = genai.GenerativeModel(
        SYNTHESIZER_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0.2),
    )
    response = model.generate_content(context)
    answer = response.text.strip()

    chunks = state.get("trimmed_chunks", [])
    source_docs = {c.get("source_doc", "") for c in chunks}
    cited = [s for s in source_docs if s and s.lower() in answer.lower()]

    if not cited and chunks:
        cited = list(source_docs)
    
    logger.info("Synthesizer: response=%d chars, cited=%s", len(answer), cited)

    return {
        "response": answer,
        "cited_sources": cited,
    }