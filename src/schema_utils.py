"""
Schema Extraction Utilities
============================
Extracts CREATE TABLE statements from SQLite databases for RAG retrieval.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """Extracts and formats database schemas for RAG indexing."""

    def __init__(self, db_path: Path):
        """
        Initialize schema extractor.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

    def extract_all_schemas(self) -> str:
        """
        Extract all CREATE TABLE statements from the database.

        Returns:
            String containing all CREATE TABLE statements separated by newlines
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Get all table creation SQL
            cursor.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )

            schemas = []
            for row in cursor.fetchall():
                if row[0]:  # Filter None values
                    schemas.append(row[0])

            conn.close()

            if not schemas:
                logger.warning(f"No tables found in {self.db_path}")
                return ""

            return "\n\n".join(schemas)

        except sqlite3.Error as e:
            logger.error(f"SQLite error extracting schemas: {e}")
            raise

    def extract_table_schema(self, table_name: str) -> Optional[str]:
        """
        Extract CREATE TABLE statement for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            CREATE TABLE statement or None if table not found
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT sql
                FROM sqlite_master
                WHERE type='table'
                AND name = ?
                """,
                (table_name,)
            )

            result = cursor.fetchone()
            conn.close()

            return result[0] if result else None

        except sqlite3.Error as e:
            logger.error(f"Error extracting schema for table {table_name}: {e}")
            return None

    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database.

        Returns:
            List of table names
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )

            tables = [row[0] for row in cursor.fetchall()]
            conn.close()

            return tables

        except sqlite3.Error as e:
            logger.error(f"Error getting table names: {e}")
            return []

    def get_table_info(self, table_name: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Get detailed column information for a table.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with column information
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            conn.close()

            return {
                "columns": [
                    {
                        "name": col[1],
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default": col[4],
                        "primary_key": bool(col[5])
                    }
                    for col in columns
                ]
            }

        except sqlite3.Error as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            return {"columns": []}


def extract_schemas_from_directory(db_dir: Path) -> Dict[str, str]:
    """
    Extract schemas from all SQLite databases in a directory.

    Args:
        db_dir: Directory containing .sqlite or .db files

    Returns:
        Dictionary mapping database names to their schemas
    """
    db_dir = Path(db_dir)
    schemas = {}

    for db_path in db_dir.rglob("*.sqlite"):
        try:
            extractor = SchemaExtractor(db_path)
            db_name = db_path.stem
            schemas[db_name] = extractor.extract_all_schemas()
            logger.info(f"Extracted schema for database: {db_name}")
        except Exception as e:
            logger.error(f"Failed to extract schema from {db_path}: {e}")

    # Also check for .db files
    for db_path in db_dir.rglob("*.db"):
        try:
            extractor = SchemaExtractor(db_path)
            db_name = db_path.stem
            if db_name not in schemas:  # Avoid duplicates
                schemas[db_name] = extractor.extract_all_schemas()
                logger.info(f"Extracted schema for database: {db_name}")
        except Exception as e:
            logger.error(f"Failed to extract schema from {db_path}: {e}")

    return schemas
