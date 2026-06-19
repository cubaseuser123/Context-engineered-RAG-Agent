"""
token_utils - Token counting and truncation using tiktoken.
"""
import logging 
import tiktoken 

logger = logging.getLogger(__name__)

encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if not text:
        return 0 

    tokens = encoder.encode(text)
    return len(tokens)

def count_messages(messages: list[dict]) -> int:
    """Count total tokens accross the list of message dicts [{role, content}]."""
    total = 0 
    for msg in messages:
        total += count_tokens(msg.get("role", ""))
        total += count_tokens(msg.get("content",""))

        #this is being added to account for formatting markers
        total += 4
    return total 

def count_entries(entries: list[dict], key: str = "fact") -> int:
    """Count total tokens across list of dicts by a specific text key."""
    total = 0 
    for entry in entries:
        text_to_count = str(entry.get(key, ""))
        total += count_tokens(text_to_count)
    return total 

def count_chunks(chunks: list[dict]) -> int:
    """Count total tokens across retrieved chunks."""
    total = 0 
    for chunk in chunks:
        content = chunk.get("content", "")
        total += count_tokens(content)
    return total 

def truncate_to_budget(text: str, budget: int) -> str:
    """Truncate text to fit with a token budget."""
    if not text:
        return text 

    tokens = encoder.encode(text)
    if len(tokens) <= budget:
        return text 

    allowed_tokens = tokens[:budget]
    truncated_text = encoder.decode(allowed_tokens)

    logger.debug("Truncated text from %d to %d tokens", len(tokens), budget)

    return truncated_text