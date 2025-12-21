"""
Tool definitions for the Insights agent.

Each tool wraps existing CLI functionality and exposes it to the LLM.
"""

from typing import Any
import subprocess
import json
import os

# Get app dir from environment or use default
APP_DIR = os.environ.get("INSIGHTS_APP_DIR", os.path.expanduser("~/.insights"))


def _run_cli(*args: str) -> dict[str, Any]:
    """Run an insights CLI command and return structured result."""
    cmd = ["uv", "run", "insights", "--app-dir", APP_DIR, *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip() if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Tool Implementations
# =============================================================================


def ingest_source(source: str, backend: str | None = None, refresh: bool = False) -> dict:
    """Ingest a file, URL, or YouTube video into the database."""
    args = ["ingest", source]
    if backend:
        args.extend(["--backend", backend])
    if refresh:
        args.append("--refresh")
    return _run_cli(*args)


def list_sources(kind: str | None = None) -> dict:
    """List all ingested sources, optionally filtered by kind. Includes descriptions for semantic matching."""
    import sqlite3
    db_path = os.path.join(APP_DIR, "insights.db")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        query = "SELECT id, kind, locator, title, description, updated_at FROM sources"
        params = []
        
        if kind:
            query += " WHERE kind = ?"
            params.append(kind)
        
        query += " ORDER BY updated_at DESC"
        
        cur.execute(query, params)
        sources = [dict(row) for row in cur.fetchall()]
        conn.close()
        
        return {
            "success": True,
            "sources": sources,
            "count": len(sources),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_source_exists(locator: str) -> dict:
    """Check if a source (URL, file path, or partial match) exists in the database. Searches locator, title, and description."""
    import sqlite3
    db_path = os.path.join(APP_DIR, "insights.db")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, kind, locator, title, description, updated_at 
            FROM sources
        """)
        sources = cur.fetchall()
        conn.close()
        
        matches = []
        locator_lower = locator.lower()
        
        for src in sources:
            src_dict = dict(src)
            src_locator = (src_dict.get("locator") or "").lower()
            src_title = (src_dict.get("title") or "").lower()
            src_desc = (src_dict.get("description") or "").lower()
            
            # Check for match in locator, title, or description
            if (locator_lower in src_locator or 
                locator_lower in src_title or 
                locator_lower in src_desc):
                matches.append(src_dict)
        
        return {
            "success": True,
            "exists": len(matches) > 0,
            "matches": matches,
            "match_count": len(matches),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_conversations(source: str | None = None, limit: int = 10) -> dict:
    """List conversations, optionally filtered to a specific source."""
    args = ["conversations", "--json", "--limit", str(limit)]
    if source:
        args.extend(["--source", source])
    result = _run_cli(*args)
    if result["success"] and result["stdout"]:
        try:
            result["conversations"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    return result


def semantic_search_sources(query: str, kind: str | None = None) -> dict:
    """
    Search sources by semantic query. Returns all sources with their descriptions
    so the LLM can match the query to the most relevant source.
    
    This is the preferred tool when looking for sources by topic/content rather than exact URL.
    """
    import sqlite3
    db_path = os.path.join(APP_DIR, "insights.db")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        sql = """
            SELECT id, kind, locator, title, description, updated_at 
            FROM sources
        """
        params = []
        
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        
        sql += " ORDER BY updated_at DESC"
        
        cur.execute(sql, params)
        sources = [dict(row) for row in cur.fetchall()]
        conn.close()
        
        return {
            "success": True,
            "query": query,
            "sources": sources,
            "count": len(sources),
            "note": "Use the title and description fields to find the best semantic match for the query.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_conversations(query: str, source: str | None = None) -> dict:
    """
    Search conversations by content/topic.
    Returns conversations whose messages or titles match the query.
    """
    # First get conversations (optionally for a source)
    convos_result = list_conversations(source=source, limit=50)
    if not convos_result["success"]:
        return convos_result

    conversations = convos_result.get("conversations", [])
    query_lower = query.lower()
    matches = []

    for convo in conversations:
        # Check title
        title = convo.get("title", "").lower()
        excerpt = convo.get("excerpt", "").lower()
        
        if query_lower in title or query_lower in excerpt:
            matches.append({
                "id": convo.get("id"),
                "title": convo.get("title"),
                "excerpt": convo.get("excerpt"),
                "updated_at": convo.get("updated_at"),
                "match_reason": "title" if query_lower in title else "content",
            })

    return {
        "success": True,
        "query": query,
        "matches": matches,
        "match_count": len(matches),
    }


def ask_source(source: str, question: str, provider: str = "anthropic") -> dict:
    """Ask a one-off question about a source."""
    args = ["ask", "-s", source, question, "--provider", provider]
    return _run_cli(*args)


def get_conversation_info(conversation_id: str) -> dict:
    """Get details about a specific conversation."""
    # Use sqlite to get conversation details directly
    import sqlite3
    db_path = os.path.join(APP_DIR, "insights.db")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Get conversation
        cur.execute("""
            SELECT id, title, created_at, updated_at 
            FROM conversations WHERE id = ?
        """, (conversation_id,))
        convo = cur.fetchone()
        
        if not convo:
            return {"success": False, "error": "Conversation not found"}
        
        # Get associated sources
        cur.execute("""
            SELECT s.id, s.kind, s.locator, s.title
            FROM sources s
            JOIN conversation_sources cs ON s.id = cs.source_id
            WHERE cs.conversation_id = ?
        """, (conversation_id,))
        sources = [dict(row) for row in cur.fetchall()]
        
        conn.close()
        
        return {
            "success": True,
            "conversation": dict(convo),
            "sources": sources,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Tool Schemas (Anthropic format)
# =============================================================================

TOOL_SCHEMAS = [
    {
        "name": "ingest_source",
        "description": "Ingest a file (PDF, DOCX, etc.), web page URL, or YouTube video URL into the database. This extracts and caches the text content for later querying.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "File path, web URL, or YouTube URL to ingest"
                },
                "backend": {
                    "type": "string",
                    "enum": ["docling", "firecrawl"],
                    "description": "Backend to use for web pages. Default is docling."
                },
                "refresh": {
                    "type": "boolean",
                    "description": "Force re-ingestion even if already cached"
                }
            },
            "required": ["source"]
        }
    },
    {
        "name": "list_sources",
        "description": "List all sources that have been ingested into the database. Can filter by type (file, url, youtube).",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["file", "url", "youtube"],
                    "description": "Filter sources by type"
                }
            }
        }
    },
    {
        "name": "check_source_exists",
        "description": "Check if a specific source exists in the database. Searches by URL, file path, title, or description (partial match).",
        "input_schema": {
            "type": "object",
            "properties": {
                "locator": {
                    "type": "string",
                    "description": "URL, file path, or search term to look for"
                }
            },
            "required": ["locator"]
        }
    },
    {
        "name": "semantic_search_sources",
        "description": "Search for sources by topic or content using natural language. Returns all sources with their descriptions so you can find the best semantic match. Use this when looking for sources by what they're ABOUT rather than exact URL/path. Example: 'video about someone who hates IDEs' or 'document about biotech AI project'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you're looking for"
                },
                "kind": {
                    "type": "string",
                    "enum": ["file", "url", "youtube"],
                    "description": "Optional: filter to specific source type"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_conversations",
        "description": "List chat conversations, optionally filtered to a specific source. Returns conversation IDs, titles, and excerpts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source ID, URL, or file path to filter conversations"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of conversations to return (default 10)"
                }
            }
        }
    },
    {
        "name": "search_conversations",
        "description": "Search conversations by topic or content. Finds conversations whose title or message excerpts match the query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or keywords to search for"
                },
                "source": {
                    "type": "string",
                    "description": "Optional: limit search to conversations about this source"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "ask_source",
        "description": "Ask a one-off question about a source. Auto-ingests if the source isn't already in the database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source to ask about (file path, URL, or source ID)"
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask"
                },
                "provider": {
                    "type": "string",
                    "enum": ["openai", "anthropic"],
                    "description": "LLM provider to use (default: anthropic)"
                }
            },
            "required": ["source", "question"]
        }
    },
    {
        "name": "get_conversation_info",
        "description": "Get detailed information about a specific conversation, including its associated sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "The conversation ID"
                }
            },
            "required": ["conversation_id"]
        }
    }
]

# Map tool names to functions
TOOL_FUNCTIONS = {
    "ingest_source": ingest_source,
    "list_sources": list_sources,
    "check_source_exists": check_source_exists,
    "semantic_search_sources": semantic_search_sources,
    "list_conversations": list_conversations,
    "search_conversations": search_conversations,
    "ask_source": ask_source,
    "get_conversation_info": get_conversation_info,
}
