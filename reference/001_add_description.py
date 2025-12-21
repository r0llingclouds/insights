"""
Migration: Add description column to sources table.

This migration adds a 'description' column to store LLM-generated
one-liner descriptions for semantic matching.

Usage:
    # Dry run (show what would happen)
    python migrations/001_add_description.py --dry-run
    
    # Run migration
    python migrations/001_add_description.py
    
    # With custom app dir
    INSIGHTS_APP_DIR=/tmp/insights-test python migrations/001_add_description.py
"""

import sqlite3
import os
import sys
from pathlib import Path

# Get app dir from environment or use default
APP_DIR = Path(os.environ.get("INSIGHTS_APP_DIR", os.path.expanduser("~/.insights")))
DB_PATH = APP_DIR / "insights.db"


def check_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column already exists in a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate(dry_run: bool = False) -> bool:
    """
    Add description column to sources table.
    
    Args:
        dry_run: If True, only print what would happen
        
    Returns:
        True if migration was applied (or would be), False if already done
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run 'insights ingest' first to create the database.")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check if already migrated
    if check_column_exists(conn, "sources", "description"):
        print("✓ Column 'description' already exists. Nothing to do.")
        conn.close()
        return False
    
    if dry_run:
        print(f"Would add 'description' column to sources table in {DB_PATH}")
        conn.close()
        return True
    
    # Run migration
    print(f"Adding 'description' column to sources table...")
    
    try:
        conn.execute("ALTER TABLE sources ADD COLUMN description TEXT")
        conn.commit()
        print("✓ Migration complete!")
        
        # Show current state
        cursor = conn.execute("SELECT COUNT(*) FROM sources")
        count = cursor.fetchone()[0]
        print(f"  {count} sources in database (all have NULL descriptions)")
        print(f"  Run 'python backfill_descriptions.py' to generate descriptions")
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        conn.rollback()
        conn.close()
        return False
    
    conn.close()
    return True


def rollback(dry_run: bool = False) -> bool:
    """
    Remove description column (SQLite doesn't support DROP COLUMN easily,
    so this recreates the table).
    
    Args:
        dry_run: If True, only print what would happen
        
    Returns:
        True if rollback was applied
    """
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    
    if not check_column_exists(conn, "sources", "description"):
        print("✓ Column 'description' doesn't exist. Nothing to rollback.")
        conn.close()
        return False
    
    if dry_run:
        print("Would remove 'description' column from sources table")
        print("WARNING: This will delete all description data!")
        conn.close()
        return True
    
    print("Removing 'description' column from sources table...")
    print("WARNING: This will delete all description data!")
    
    try:
        # SQLite workaround: recreate table without the column
        # First, get current schema without description
        cursor = conn.execute("PRAGMA table_info(sources)")
        columns = [row[1] for row in cursor.fetchall() if row[1] != "description"]
        columns_str = ", ".join(columns)
        
        conn.executescript(f"""
            BEGIN TRANSACTION;
            
            CREATE TABLE sources_backup AS 
            SELECT {columns_str} FROM sources;
            
            DROP TABLE sources;
            
            CREATE TABLE sources AS 
            SELECT * FROM sources_backup;
            
            DROP TABLE sources_backup;
            
            COMMIT;
        """)
        
        print("✓ Rollback complete!")
        
    except Exception as e:
        print(f"✗ Rollback failed: {e}")
        conn.rollback()
        conn.close()
        return False
    
    conn.close()
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add description column to sources table")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--rollback", action="store_true", help="Remove the column instead")
    
    args = parser.parse_args()
    
    print(f"Database: {DB_PATH}")
    print()
    
    if args.rollback:
        success = rollback(dry_run=args.dry_run)
    else:
        success = migrate(dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)
