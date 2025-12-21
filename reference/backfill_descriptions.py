"""
Backfill descriptions for existing sources.

This script generates one-liner descriptions for all sources that
don't have one yet. It reads the cached content from the database
and uses an LLM to generate descriptions.

Usage:
    # Backfill all sources without descriptions
    python backfill_descriptions.py
    
    # Dry run (show what would be processed)
    python backfill_descriptions.py --dry-run
    
    # Limit to N sources (useful for testing)
    python backfill_descriptions.py --limit 5
    
    # Force regenerate all descriptions
    python backfill_descriptions.py --force
    
    # With custom app dir
    INSIGHTS_APP_DIR=/tmp/insights-test python backfill_descriptions.py
"""

import sqlite3
import os
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import insights modules
sys.path.insert(0, str(Path(__file__).parent))

from insights.describe import generate_description

# Get app dir from environment or use default
APP_DIR = Path(os.environ.get("INSIGHTS_APP_DIR", os.path.expanduser("~/.insights")))
DB_PATH = APP_DIR / "insights.db"


def get_sources_needing_descriptions(
    conn: sqlite3.Connection,
    force: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """
    Get sources that need descriptions generated.
    
    Args:
        conn: Database connection
        force: If True, get all sources (even those with descriptions)
        limit: Max number of sources to return
        
    Returns:
        List of source dicts with id, kind, locator, title
    """
    conn.row_factory = sqlite3.Row
    
    if force:
        query = "SELECT id, kind, locator, title FROM sources ORDER BY updated_at DESC"
    else:
        query = "SELECT id, kind, locator, title FROM sources WHERE description IS NULL ORDER BY updated_at DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor = conn.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def get_source_content(conn: sqlite3.Connection, source_id: str) -> str | None:
    """
    Get the cached content for a source.
    
    Note: Adjust this query based on your actual cache table schema.
    """
    # Try common table names for cached content
    for table in ["cache", "source_content", "content"]:
        try:
            cursor = conn.execute(
                f"SELECT content FROM {table} WHERE source_id = ?",
                (source_id,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
        except sqlite3.OperationalError:
            continue
    
    # Alternative: content might be in sources table directly
    try:
        cursor = conn.execute(
            "SELECT content FROM sources WHERE id = ?",
            (source_id,)
        )
        row = cursor.fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    
    return None


def update_description(conn: sqlite3.Connection, source_id: str, description: str) -> None:
    """Update the description for a source."""
    conn.execute(
        "UPDATE sources SET description = ? WHERE id = ?",
        (description, source_id)
    )


def backfill(
    dry_run: bool = False,
    force: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    """
    Generate descriptions for sources that need them.
    
    Args:
        dry_run: If True, only print what would happen
        force: If True, regenerate all descriptions
        limit: Max number of sources to process
        
    Returns:
        Tuple of (processed_count, error_count)
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return 0, 0
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check if description column exists
    cursor = conn.execute("PRAGMA table_info(sources)")
    columns = [row[1] for row in cursor.fetchall()]
    if "description" not in columns:
        print("Error: 'description' column doesn't exist.")
        print("Run the migration first: python migrations/001_add_description.py")
        conn.close()
        return 0, 0
    
    # Get sources to process
    sources = get_sources_needing_descriptions(conn, force=force, limit=limit)
    
    if not sources:
        print("✓ All sources already have descriptions. Nothing to do.")
        conn.close()
        return 0, 0
    
    print(f"Found {len(sources)} sources to process")
    
    if dry_run:
        print("\nWould generate descriptions for:")
        for src in sources:
            print(f"  [{src['kind']}] {src['title'] or src['locator'][:60]}")
        conn.close()
        return len(sources), 0
    
    print()
    
    processed = 0
    errors = 0
    
    for i, src in enumerate(sources, 1):
        source_id = src["id"]
        display_name = src["title"] or src["locator"][:50]
        
        print(f"[{i}/{len(sources)}] {display_name}...", end=" ", flush=True)
        
        # Get content
        content = get_source_content(conn, source_id)
        
        if not content:
            print("⚠ No content found, skipping")
            errors += 1
            continue
        
        try:
            # Generate description
            description = generate_description(content)
            
            # Update database
            update_description(conn, source_id, description)
            conn.commit()
            
            print(f"✓")
            print(f"   → {description}")
            
            processed += 1
            
            # Small delay to avoid rate limits
            time.sleep(0.2)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            errors += 1
    
    conn.close()
    
    print()
    print(f"Done! Processed: {processed}, Errors: {errors}")
    
    return processed, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate descriptions for sources without them"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate descriptions even for sources that have them"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sources to process"
    )
    
    args = parser.parse_args()
    
    print(f"Database: {DB_PATH}")
    print()
    
    processed, errors = backfill(
        dry_run=args.dry_run,
        force=args.force,
        limit=args.limit,
    )
    
    sys.exit(0 if errors == 0 else 1)
