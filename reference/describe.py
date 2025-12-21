"""
Generate one-liner descriptions for ingested sources.

This module provides a lightweight way to add semantic searchability
to sources without requiring a vector database. The LLM agent can
match user queries against these descriptions.

Usage:
    from insights.describe import generate_description
    
    description = generate_description(content)
    # "Developer explains mass exodus from traditional IDEs to Vim/terminal workflows"
"""

from anthropic import Anthropic

# Using Haiku for speed and cost efficiency
DEFAULT_MODEL = "claude-haiku-4-5-20250929"

DESCRIBE_SYSTEM = """\
You generate concise, searchable one-liner descriptions of content.
Your descriptions should:
- Be under 150 characters
- Capture the main topic, thesis, or key takeaway
- Use specific terms someone might search for
- Avoid generic phrases like "discusses various topics" or "talks about things"

Respond with ONLY the description, no quotes or prefixes."""

DESCRIBE_USER = """\
Generate a one-liner description for this content:

---
{content}
---

One-liner:"""


def generate_description(
    content: str,
    max_content_chars: int = 8000,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Generate a short, searchable description of the content.
    
    Args:
        content: The full text content of the source
        max_content_chars: Max chars to send to the LLM (truncates if longer)
        model: Anthropic model to use (default: Haiku for speed/cost)
        
    Returns:
        A one-liner description (max ~150 chars)
        
    Example:
        >>> generate_description("In this video I explain why I stopped using VS Code...")
        "Developer explains mass exodus from traditional IDEs to Vim/terminal workflows"
    """
    client = Anthropic()
    
    # Truncate content to save tokens, but try to keep meaningful parts
    if len(content) > max_content_chars:
        # Take beginning and end for better coverage
        half = max_content_chars // 2
        truncated = content[:half] + "\n\n[...content truncated...]\n\n" + content[-half:]
    else:
        truncated = content
    
    response = client.messages.create(
        model=model,
        max_tokens=100,
        system=DESCRIBE_SYSTEM,
        messages=[{
            "role": "user",
            "content": DESCRIBE_USER.format(content=truncated)
        }]
    )
    
    description = response.content[0].text.strip()
    
    # Clean up any quotes the model might add
    description = description.strip('"\'')
    
    # Ensure it's not too long
    if len(description) > 200:
        description = description[:197] + "..."
    
    return description


def generate_description_batch(
    contents: list[tuple[str, str]],
    max_content_chars: int = 8000,
    model: str = DEFAULT_MODEL,
) -> list[tuple[str, str]]:
    """
    Generate descriptions for multiple sources.
    
    Args:
        contents: List of (source_id, content) tuples
        max_content_chars: Max chars per source
        model: Anthropic model to use
        
    Returns:
        List of (source_id, description) tuples
        
    Note:
        This processes sequentially. For large batches, consider
        using asyncio or the Anthropic batch API.
    """
    results = []
    
    for source_id, content in contents:
        try:
            description = generate_description(
                content,
                max_content_chars=max_content_chars,
                model=model,
            )
            results.append((source_id, description))
        except Exception as e:
            # On error, use a fallback
            results.append((source_id, f"[Description generation failed: {e}]"))
    
    return results


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], "r") as f:
            content = f.read()
    else:
        # Read from stdin
        print("Paste content (Ctrl+D to finish):")
        content = sys.stdin.read()
    
    description = generate_description(content)
    print(f"\nDescription: {description}")
    print(f"Length: {len(description)} chars")
