"""
SQL Agent with RAG and Self-Correction
=======================================
Main agent class that orchestrates RAG retrieval, SQL generation,
execution, and agentic self-correction.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re

from .rag_retriever import SchemaRetriever, SimpleSchemaRetriever
from .sql_executor import SafeSQLExecutor
from .schema_utils import SchemaExtractor

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from SQL agent."""
    success: bool
    sql_query: Optional[str]
    results: Optional[Any]
    error_message: Optional[str]
    attempts: int
    retrieved_schemas: Optional[List[str]] = None


class SQLAgent:
    """
    Production-ready SQL Agent with RAG and self-correction.

    Workflow:
    1. User asks question
    2. RAG retrieves relevant schemas
    3. Model generates SQL
    4. Execute SQL on database
    5. If error → self-correct (max 3 attempts)
    6. Return results or error
    """

    def __init__(
        self,
        model,
        tokenizer,
        db_path: Path,
        schema_retriever: Optional[SchemaRetriever] = None,
        max_correction_attempts: int = 3,
        use_rag: bool = True
    ):
        """
        Initialize SQL Agent.

        Args:
            model: Fine-tuned language model (Unsloth model)
            tokenizer: Model tokenizer
            db_path: Path to SQLite database
            schema_retriever: RAG retriever (optional, created if not provided)
            max_correction_attempts: Max self-correction loops (default: 3)
            use_rag: Whether to use RAG retrieval (default: True)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = Path(db_path)
        self.max_attempts = max_correction_attempts
        self.use_rag = use_rag

        # Initialize SQL executor (read-only for safety)
        self.executor = SafeSQLExecutor(
            db_path=self.db_path,
            timeout=10.0,
            read_only=True
        )

        # Initialize schema retriever
        if schema_retriever:
            self.retriever = schema_retriever
        elif use_rag:
            self.retriever = SchemaRetriever()
        else:
            # Use simple retriever (no vector search)
            self.retriever = SimpleSchemaRetriever()

        # Extract and index schemas from database
        self._initialize_schemas()

        logger.info(f"SQLAgent initialized for database: {db_path}")

    def _initialize_schemas(self) -> None:
        """Extract and index schemas from the database."""
        try:
            extractor = SchemaExtractor(self.db_path)
            full_schema = extractor.extract_all_schemas()

            if self.use_rag and isinstance(self.retriever, SchemaRetriever):
                # Index in ChromaDB for RAG
                db_name = self.db_path.stem
                self.retriever.add_full_database_schema(db_name, full_schema)
                logger.info(f"Indexed schema in RAG retriever for: {db_name}")

            elif isinstance(self.retriever, SimpleSchemaRetriever):
                # Store directly
                db_name = self.db_path.stem
                self.retriever.add_schema(db_name, full_schema)
                logger.info(f"Loaded schema for: {db_name}")

        except Exception as e:
            logger.error(f"Failed to initialize schemas: {e}")
            raise

    def query(
        self,
        question: str,
        temperature: float = 0.1,
        max_new_tokens: int = 256
    ) -> AgentResponse:
        """
        Main query method with self-correction.

        Args:
            question: Natural language question
            temperature: Model temperature (default: 0.1 for deterministic SQL)
            max_new_tokens: Max tokens to generate

        Returns:
            AgentResponse with results or error
        """
        logger.info(f"Processing question: {question}")

        # Step 1: Retrieve relevant schemas
        schemas = self._retrieve_schemas(question)

        if not schemas:
            return AgentResponse(
                success=False,
                sql_query=None,
                results=None,
                error_message="No database schema available",
                attempts=0
            )

        # Step 2: Generate SQL with self-correction loop
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"Attempt {attempt}/{self.max_attempts}")

            # Generate SQL
            sql_query = self._generate_sql(
                question=question,
                schemas=schemas,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                error_context=None if attempt == 1 else None  # Will add error feedback
            )

            if not sql_query:
                return AgentResponse(
                    success=False,
                    sql_query=None,
                    results=None,
                    error_message="Failed to generate SQL",
                    attempts=attempt,
                    retrieved_schemas=schemas
                )

            # Clean SQL
            sql_query = self._clean_sql(sql_query)

            # Step 3: Execute SQL
            success, results, error_msg = self.executor.execute_query(sql_query)

            if success:
                # Success! Return results
                logger.info(f"Query succeeded on attempt {attempt}")
                return AgentResponse(
                    success=True,
                    sql_query=sql_query,
                    results=results,
                    error_message=None,
                    attempts=attempt,
                    retrieved_schemas=schemas
                )

            else:
                # Error occurred
                logger.warning(f"Attempt {attempt} failed: {error_msg}")

                if attempt < self.max_attempts:
                    # Try to self-correct
                    logger.info("Attempting self-correction...")
                    sql_query = self._self_correct(
                        question=question,
                        schemas=schemas,
                        failed_sql=sql_query,
                        error_message=error_msg,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens
                    )

                    if sql_query:
                        # Execute corrected SQL
                        sql_query = self._clean_sql(sql_query)
                        success, results, error_msg = self.executor.execute_query(sql_query)

                        if success:
                            logger.info(f"Self-correction succeeded on attempt {attempt}")
                            return AgentResponse(
                                success=True,
                                sql_query=sql_query,
                                results=results,
                                error_message=None,
                                attempts=attempt,
                                retrieved_schemas=schemas
                            )

        # All attempts failed
        logger.error(f"All {self.max_attempts} attempts failed")
        return AgentResponse(
            success=False,
            sql_query=sql_query,
            results=None,
            error_message=f"Failed after {self.max_attempts} attempts. Last error: {error_msg}",
            attempts=self.max_attempts,
            retrieved_schemas=schemas
        )

    def _retrieve_schemas(self, question: str) -> List[str]:
        """
        Retrieve relevant schemas using RAG or simple retriever.

        Args:
            question: User question

        Returns:
            List of schema strings
        """
        if self.use_rag and isinstance(self.retriever, SchemaRetriever):
            # Use RAG retrieval
            db_name = self.db_path.stem
            retrieved = self.retriever.retrieve_schemas(
                question=question,
                database_name=db_name,
                top_k=1  # Usually just need the full database schema
            )

            return [item['schema'] for item in retrieved]

        elif isinstance(self.retriever, SimpleSchemaRetriever):
            # Use simple retrieval
            db_name = self.db_path.stem
            schema = self.retriever.retrieve_schema(db_name)
            return [schema] if schema else []

        else:
            return []

    def _generate_sql(
        self,
        question: str,
        schemas: List[str],
        temperature: float,
        max_new_tokens: int,
        error_context: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate SQL using the fine-tuned model.

        Args:
            question: User question
            schemas: List of relevant schemas
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            error_context: Previous error for self-correction (optional)

        Returns:
            Generated SQL query or None
        """
        # Format prompt using Llama-3 chat template
        schema_text = "\n\n".join(schemas)

        instruction = (
            "You are an expert SQL generator. Given a database schema and a natural language question, "
            "generate the correct SQL query to answer the question. "
            "Output only the SQL query without any explanations."
        )

        if error_context:
            instruction += (
                f"\n\nThe previous SQL query failed with this error:\n{error_context}\n"
                "Please fix the query."
            )

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

Database Schema:
{schema_text}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        try:
            # Tokenize
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode
            generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract SQL (after "assistant" token)
            sql = generated.split("assistant")[-1].strip()

            return sql

        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return None

    def _self_correct(
        self,
        question: str,
        schemas: List[str],
        failed_sql: str,
        error_message: str,
        temperature: float,
        max_new_tokens: int
    ) -> Optional[str]:
        """
        Attempt to self-correct failed SQL.

        Args:
            question: Original question
            schemas: Database schemas
            failed_sql: SQL that failed
            error_message: Error from failed execution
            temperature: Sampling temperature
            max_new_tokens: Max tokens

        Returns:
            Corrected SQL or None
        """
        logger.info("Generating corrected SQL...")

        error_context = f"Failed SQL: {failed_sql}\nError: {error_message}"

        return self._generate_sql(
            question=question,
            schemas=schemas,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            error_context=error_context
        )

    def _clean_sql(self, sql: str) -> str:
        """
        Clean generated SQL query.

        Args:
            sql: Raw SQL from model

        Returns:
            Cleaned SQL
        """
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)

        # Remove extra whitespace
        sql = ' '.join(sql.split())

        # Ensure ends with semicolon
        sql = sql.rstrip()
        if not sql.endswith(';'):
            sql += ';'

        return sql

    def get_schema_preview(self) -> str:
        """
        Get preview of database schema.

        Returns:
            Schema as string
        """
        try:
            extractor = SchemaExtractor(self.db_path)
            return extractor.extract_all_schemas()
        except Exception as e:
            logger.error(f"Error getting schema preview: {e}")
            return "Schema not available"
