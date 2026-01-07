"""
RAG Schema Retriever
====================
Retrieves relevant database schemas using semantic search with ChromaDB.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)


class SchemaRetriever:
    """
    RAG-based schema retrieval using ChromaDB for semantic search.

    This allows the agent to retrieve only relevant table schemas
    instead of sending all schemas to the LLM (context limit issue).
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "database_schemas"
    ):
        """
        Initialize the schema retriever.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Database table schemas for SQL generation"}
        )

        logger.info(f"Initialized SchemaRetriever with collection: {collection_name}")

    def add_schema(
        self,
        schema: str,
        table_name: str,
        database_name: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a table schema to the vector database.

        Args:
            schema: CREATE TABLE statement
            table_name: Name of the table
            database_name: Name of the database
            metadata: Additional metadata (optional)
        """
        if not schema or not schema.strip():
            logger.warning(f"Empty schema for {database_name}.{table_name}, skipping")
            return

        # Create unique ID for this schema
        schema_id = self._generate_id(database_name, table_name)

        # Prepare metadata
        meta = {
            "table_name": table_name,
            "database_name": database_name,
            **(metadata or {})
        }

        try:
            # Add to ChromaDB
            self.collection.add(
                documents=[schema],
                metadatas=[meta],
                ids=[schema_id]
            )

            logger.info(f"Added schema: {database_name}.{table_name}")

        except Exception as e:
            logger.error(f"Failed to add schema {database_name}.{table_name}: {e}")
            raise

    def add_database_schemas(
        self,
        database_name: str,
        schemas: Dict[str, str]
    ) -> None:
        """
        Add all table schemas from a database.

        Args:
            database_name: Name of the database
            schemas: Dictionary mapping table names to CREATE TABLE statements
        """
        for table_name, schema in schemas.items():
            if schema:
                self.add_schema(
                    schema=schema,
                    table_name=table_name,
                    database_name=database_name
                )

    def add_full_database_schema(
        self,
        database_name: str,
        full_schema: str
    ) -> None:
        """
        Add a complete database schema as a single document.

        Useful when you want to retrieve entire database schemas
        rather than individual tables.

        Args:
            database_name: Name of the database
            full_schema: All CREATE TABLE statements concatenated
        """
        if not full_schema or not full_schema.strip():
            logger.warning(f"Empty schema for database {database_name}")
            return

        schema_id = self._generate_id(database_name, "_full_schema")

        meta = {
            "database_name": database_name,
            "type": "full_database_schema"
        }

        try:
            self.collection.add(
                documents=[full_schema],
                metadatas=[meta],
                ids=[schema_id]
            )

            logger.info(f"Added full schema for database: {database_name}")

        except Exception as e:
            logger.error(f"Failed to add full schema for {database_name}: {e}")
            raise

    def retrieve_schemas(
        self,
        question: str,
        database_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, str]]:
        """
        Retrieve relevant schemas for a question using semantic search.

        Args:
            question: Natural language question
            database_name: Optional filter by database name
            top_k: Number of schemas to retrieve

        Returns:
            List of dictionaries with schema and metadata
        """
        try:
            # Build query filter
            where_filter = None
            if database_name:
                where_filter = {"database_name": database_name}

            # Query ChromaDB
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k,
                where=where_filter
            )

            # Format results
            schemas = []
            if results and results['documents'] and results['documents'][0]:
                for i, schema in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else None

                    schemas.append({
                        "schema": schema,
                        "table_name": metadata.get("table_name", "unknown"),
                        "database_name": metadata.get("database_name", "unknown"),
                        "distance": distance,
                        "metadata": metadata
                    })

            logger.info(f"Retrieved {len(schemas)} schemas for question")
            return schemas

        except Exception as e:
            logger.error(f"Error retrieving schemas: {e}")
            return []

    def get_database_schema(self, database_name: str) -> Optional[str]:
        """
        Get the full schema for a specific database.

        Args:
            database_name: Name of the database

        Returns:
            Full schema or None if not found
        """
        schema_id = self._generate_id(database_name, "_full_schema")

        try:
            result = self.collection.get(ids=[schema_id])

            if result and result['documents']:
                return result['documents'][0]

            return None

        except Exception as e:
            logger.error(f"Error getting database schema: {e}")
            return None

    def clear_database(self, database_name: str) -> None:
        """
        Remove all schemas for a specific database.

        Args:
            database_name: Name of the database to remove
        """
        try:
            # Get all IDs for this database
            results = self.collection.get(
                where={"database_name": database_name}
            )

            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Cleared {len(results['ids'])} schemas for database: {database_name}")

        except Exception as e:
            logger.error(f"Error clearing database {database_name}: {e}")

    def list_databases(self) -> List[str]:
        """
        Get list of all databases in the collection.

        Returns:
            List of unique database names
        """
        try:
            results = self.collection.get()

            if not results or not results['metadatas']:
                return []

            databases = set()
            for metadata in results['metadatas']:
                if 'database_name' in metadata:
                    databases.add(metadata['database_name'])

            return sorted(list(databases))

        except Exception as e:
            logger.error(f"Error listing databases: {e}")
            return []

    def count_schemas(self) -> int:
        """
        Get total number of schemas in the collection.

        Returns:
            Count of schemas
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error counting schemas: {e}")
            return 0

    def _generate_id(self, database_name: str, table_name: str) -> str:
        """Generate unique ID for a schema."""
        combined = f"{database_name}::{table_name}"
        return hashlib.md5(combined.encode()).hexdigest()


class SimpleSchemaRetriever:
    """
    Simple non-RAG schema retriever for single database scenarios.

    Use this when you only have one database and want to
    inject all schemas without semantic search.
    """

    def __init__(self):
        """Initialize simple retriever."""
        self.schemas: Dict[str, str] = {}
        logger.info("Initialized SimpleSchemaRetriever")

    def add_schema(self, database_name: str, schema: str) -> None:
        """
        Add a database schema.

        Args:
            database_name: Name of the database
            schema: Full CREATE TABLE statements
        """
        self.schemas[database_name] = schema
        logger.info(f"Added schema for database: {database_name}")

    def retrieve_schema(self, database_name: str) -> Optional[str]:
        """
        Retrieve schema for a database.

        Args:
            database_name: Name of the database

        Returns:
            Schema or None if not found
        """
        return self.schemas.get(database_name)

    def retrieve_all_schemas(self) -> str:
        """
        Retrieve all schemas concatenated.

        Returns:
            All schemas joined with separators
        """
        if not self.schemas:
            return ""

        parts = []
        for db_name, schema in self.schemas.items():
            parts.append(f"-- Database: {db_name}\n{schema}")

        return "\n\n".join(parts)
