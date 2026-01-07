"""
SQL Executor
============
Executes SQL queries on SQLite databases with error handling and result formatting.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SQLExecutor:
    """Executes SQL queries on SQLite databases with comprehensive error handling."""

    def __init__(self, db_path: Path, timeout: float = 10.0):
        """
        Initialize SQL executor.

        Args:
            db_path: Path to SQLite database
            timeout: Query timeout in seconds (default: 10s)
        """
        self.db_path = Path(db_path)
        self.timeout = timeout

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        logger.info(f"Initialized SQLExecutor for: {db_path}")

    def execute_query(
        self,
        sql: str,
        fetch_results: bool = True
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute a SQL query and return results.

        Args:
            sql: SQL query to execute
            fetch_results: Whether to fetch and return results (default: True)

        Returns:
            Tuple of (success, results/affected_rows, error_message)
            - success: Boolean indicating if query succeeded
            - results: List of rows if SELECT, else number of affected rows
            - error_message: Error description if failed, else None
        """
        sql = sql.strip()

        if not sql:
            return False, None, "Empty SQL query"

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            logger.info(f"Executing SQL: {sql[:100]}...")

            # Execute query
            cursor.execute(sql)

            # Determine if this is a SELECT query
            is_select = sql.upper().strip().startswith("SELECT")

            if is_select and fetch_results:
                # Fetch all results
                rows = cursor.fetchall()

                # Convert to list of dictionaries
                results = [dict(row) for row in rows]

                logger.info(f"Query successful. Returned {len(results)} rows.")
                conn.close()

                return True, results, None

            else:
                # For INSERT/UPDATE/DELETE, return affected rows
                affected = cursor.rowcount
                conn.commit()
                conn.close()

                logger.info(f"Query successful. Affected {affected} rows.")
                return True, affected, None

        except sqlite3.Error as e:
            error_msg = self._format_error(e, sql)
            logger.error(f"SQL execution failed: {error_msg}")
            return False, None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg

    def _format_error(self, error: sqlite3.Error, sql: str) -> str:
        """
        Format SQLite error for better readability.

        Args:
            error: SQLite error
            sql: The SQL query that caused the error

        Returns:
            Formatted error message
        """
        error_str = str(error)

        # Common error patterns
        if "no such table" in error_str.lower():
            return f"Table does not exist: {error_str}"

        elif "no such column" in error_str.lower():
            return f"Column does not exist: {error_str}"

        elif "syntax error" in error_str.lower():
            return f"SQL syntax error: {error_str}\nQuery: {sql[:200]}"

        elif "ambiguous column name" in error_str.lower():
            return f"Ambiguous column (need table alias): {error_str}"

        elif "datatype mismatch" in error_str.lower():
            return f"Data type mismatch: {error_str}"

        else:
            return f"SQLite error: {error_str}"

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax without executing.

        Args:
            sql: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Use EXPLAIN to validate without executing
            cursor.execute(f"EXPLAIN {sql}")
            conn.close()

            return True, None

        except sqlite3.Error as e:
            return False, self._format_error(e, sql)

    def get_query_result_preview(
        self,
        sql: str,
        limit: int = 5
    ) -> Tuple[bool, List[Dict], Optional[str]]:
        """
        Execute query and return limited preview of results.

        Args:
            sql: SQL query
            limit: Maximum rows to return

        Returns:
            Tuple of (success, preview_results, error_message)
        """
        # Modify query to add LIMIT if not present
        sql_upper = sql.upper().strip()

        if "LIMIT" not in sql_upper:
            # Add LIMIT to the query
            if sql.rstrip().endswith(";"):
                sql = sql.rstrip()[:-1] + f" LIMIT {limit};"
            else:
                sql = sql.rstrip() + f" LIMIT {limit}"

        return self.execute_query(sql)

    def explain_query(self, sql: str) -> Tuple[bool, str, Optional[str]]:
        """
        Get query execution plan.

        Args:
            sql: SQL query to explain

        Returns:
            Tuple of (success, explanation, error_message)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan_rows = cursor.fetchall()

            conn.close()

            explanation = "\n".join([str(row) for row in plan_rows])
            return True, explanation, None

        except sqlite3.Error as e:
            return False, "", str(e)


class SafeSQLExecutor(SQLExecutor):
    """
    SQL executor with additional safety constraints.

    - Read-only mode
    - Query timeout enforcement
    - Dangerous command filtering
    """

    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
        "INSERT", "UPDATE", "REPLACE", "ATTACH", "DETACH"
    ]

    def __init__(
        self,
        db_path: Path,
        timeout: float = 5.0,
        read_only: bool = True
    ):
        """
        Initialize safe executor.

        Args:
            db_path: Path to database
            timeout: Query timeout (default: 5s for safety)
            read_only: Only allow SELECT queries (default: True)
        """
        super().__init__(db_path, timeout)
        self.read_only = read_only

        if read_only:
            logger.info("SafeSQLExecutor initialized in READ-ONLY mode")

    def execute_query(
        self,
        sql: str,
        fetch_results: bool = True
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute query with safety checks.

        Args:
            sql: SQL query
            fetch_results: Whether to fetch results

        Returns:
            Tuple of (success, results, error_message)
        """
        # Safety check: Block dangerous commands in read-only mode
        if self.read_only:
            sql_upper = sql.upper()

            for keyword in self.DANGEROUS_KEYWORDS:
                if keyword in sql_upper:
                    error_msg = (
                        f"Dangerous command '{keyword}' blocked in read-only mode. "
                        f"Only SELECT queries allowed."
                    )
                    logger.warning(error_msg)
                    return False, None, error_msg

        # Execute with parent class method
        return super().execute_query(sql, fetch_results)
