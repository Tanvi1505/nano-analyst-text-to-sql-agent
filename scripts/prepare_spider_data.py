"""
Spider Dataset Preparation for Nano-Analyst SQL Agent
======================================================
Downloads and formats the Yale Spider dataset for QLoRA fine-tuning with Unsloth.

This script:
1. Downloads Spider dataset (Text-to-SQL benchmark)
2. Parses schema information from databases
3. Formats data into Alpaca instruction format
4. Creates train/validation splits
5. Exports JSON compatible with Unsloth training

Author: Senior AI Architect
Dataset: Spider (Yale) - https://yale-lily.github.io/spider
"""

import json
import sqlite3
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SpiderExample:
    """Structured representation of a Spider training example."""
    question: str
    database_id: str
    query: str
    schema: str


class SpiderDatasetProcessor:
    """
    Processes Spider dataset into instruction-tuning format for LLM fine-tuning.

    The Spider dataset contains:
    - Natural language questions
    - SQL queries (labels)
    - Database schemas
    - SQLite database files
    """

    SPIDER_URL = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"

    def __init__(self, base_dir: Path):
        """
        Initialize the processor.

        Args:
            base_dir: Root directory for data storage (e.g., ~/nano-analyst/data)
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw" / "spider"
        self.processed_dir = self.base_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_spider(self) -> None:
        """
        Download Spider dataset from official source.

        Note: Spider is publicly available but hosted on Google Drive.
        For automated downloads, you may need to use gdown or manual download.
        """
        logger.info("=" * 70)
        logger.info("SPIDER DATASET DOWNLOAD")
        logger.info("=" * 70)
        logger.info(
            "\nFor automated download, install: pip install gdown\n"
            "Then run: gdown 1TqleXec_OykOYFREKKtschzY29dUcVAQ\n"
            "\nAlternatively, manually download from:"
            "\nhttps://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
        )

        zip_path = self.raw_dir / "spider.zip"

        if zip_path.exists():
            logger.info(f"Found existing spider.zip at {zip_path}")
        else:
            logger.info(f"Please download Spider and place at: {zip_path}")
            logger.info("Attempting automated download with gdown...")

            try:
                import gdown
                gdown.download(
                    "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ",
                    str(zip_path),
                    quiet=False
                )
            except ImportError:
                logger.error("gdown not installed. Install with: pip install gdown")
                raise

        # Extract
        logger.info(f"Extracting to {self.raw_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        logger.info("Extraction complete!")

    def extract_schema_from_sqlite(self, db_path: Path) -> str:
        """
        Extract CREATE TABLE statements from SQLite database.

        Args:
            db_path: Path to .sqlite database file

        Returns:
            String containing all CREATE TABLE statements
        """
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get all table creation SQL
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )

            schemas = []
            for row in cursor.fetchall():
                if row[0]:  # Filter None values
                    schemas.append(row[0])

            conn.close()
            return "\n\n".join(schemas)

        except Exception as e:
            logger.error(f"Error extracting schema from {db_path}: {e}")
            return ""

    def format_as_alpaca_instruction(
        self,
        example: SpiderExample,
        include_schema: bool = True
    ) -> Dict[str, str]:
        """
        Format example into Alpaca instruction format for Unsloth.

        Alpaca Format:
        - instruction: The task description (constant for all examples)
        - input: The question + schema context
        - output: The SQL query

        Args:
            example: SpiderExample instance
            include_schema: Whether to include schema in the input

        Returns:
            Dictionary with instruction/input/output keys
        """
        instruction = (
            "You are an expert SQL generator. Given a database schema and a natural language question, "
            "generate the correct SQL query to answer the question. "
            "Output only the SQL query without any explanations."
        )

        if include_schema:
            input_text = (
                f"Database Schema:\n{example.schema}\n\n"
                f"Question: {example.question}"
            )
        else:
            input_text = f"Question: {example.question}"

        return {
            "instruction": instruction,
            "input": input_text,
            "output": example.query
        }

    def load_spider_split(self, split: str = "train") -> List[SpiderExample]:
        """
        Load Spider train or dev split.

        Args:
            split: Either 'train' or 'dev'

        Returns:
            List of SpiderExample objects
        """
        spider_dir = self.raw_dir / "spider"

        # Load the JSON file
        json_path = spider_dir / f"{split}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Spider {split}.json not found at {json_path}. "
                f"Did you download and extract Spider?"
            )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load table schemas (Spider provides tables.json with schema info)
        tables_path = spider_dir / "tables.json"
        with open(tables_path, 'r', encoding='utf-8') as f:
            tables_data = json.load(f)

        # Create database_id -> schema mapping
        db_schemas = {}
        database_dir = spider_dir / "database"

        for table_info in tables_data:
            db_id = table_info['db_id']

            # Try to extract from actual SQLite file (more accurate)
            db_path = database_dir / db_id / f"{db_id}.sqlite"

            if db_path.exists():
                schema = self.extract_schema_from_sqlite(db_path)
            else:
                # Fallback: construct from tables.json
                schema = self._construct_schema_from_json(table_info)

            db_schemas[db_id] = schema

        # Parse examples
        examples = []
        for item in data:
            question = item['question']
            db_id = item['db_id']
            query = item['query']

            schema = db_schemas.get(db_id, "")

            if not schema:
                logger.warning(f"No schema found for database: {db_id}")
                continue

            examples.append(SpiderExample(
                question=question,
                database_id=db_id,
                query=query,
                schema=schema
            ))

        logger.info(f"Loaded {len(examples)} examples from {split} split")
        return examples

    def _construct_schema_from_json(self, table_info: Dict) -> str:
        """
        Fallback: Construct schema from Spider's tables.json format.

        Args:
            table_info: Dictionary from tables.json

        Returns:
            String representation of schema
        """
        schema_lines = []

        table_names = table_info['table_names_original']
        column_names = table_info['column_names_original']
        column_types = table_info['column_types']

        # Group columns by table
        table_columns = {i: [] for i in range(len(table_names))}

        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:  # Skip if no table association
                continue
            col_type = column_types[col_idx]
            table_columns[table_idx].append(f"{col_name} {col_type}")

        # Generate CREATE TABLE statements
        for table_idx, table_name in enumerate(table_names):
            columns = table_columns.get(table_idx, [])
            if columns:
                cols_str = ",\n  ".join(columns)
                schema_lines.append(
                    f"CREATE TABLE {table_name} (\n  {cols_str}\n);"
                )

        return "\n\n".join(schema_lines)

    def create_train_val_split(
        self,
        examples: List[SpiderExample],
        val_ratio: float = 0.1
    ) -> Tuple[List[SpiderExample], List[SpiderExample]]:
        """
        Split data into training and validation sets.

        Args:
            examples: List of all examples
            val_ratio: Fraction for validation (default 10%)

        Returns:
            Tuple of (train_examples, val_examples)
        """
        import random
        random.seed(42)  # Reproducibility

        shuffled = examples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - val_ratio))
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

        logger.info(f"Split: {len(train)} train, {len(val)} validation")
        return train, val

    def save_for_unsloth(
        self,
        examples: List[SpiderExample],
        output_path: Path,
        include_schema: bool = True
    ) -> None:
        """
        Save formatted data as JSON for Unsloth training.

        Args:
            examples: List of SpiderExample objects
            output_path: Path to save JSON file
            include_schema: Whether to include schema in input
        """
        formatted_data = [
            self.format_as_alpaca_instruction(ex, include_schema)
            for ex in examples
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(formatted_data)} examples to {output_path}")

    def process_full_pipeline(self) -> None:
        """
        Execute the complete data preparation pipeline.

        Steps:
        1. Download Spider dataset
        2. Load train split
        3. Create internal train/val split
        4. Format as Alpaca instructions
        5. Save for Unsloth training
        """
        logger.info("\n" + "=" * 70)
        logger.info("NANO-ANALYST DATA PREPARATION PIPELINE")
        logger.info("=" * 70 + "\n")

        # Step 1: Download
        logger.info("[1/5] Downloading Spider dataset...")
        try:
            self.download_spider()
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info("Please manually download and extract Spider to:")
            logger.info(str(self.raw_dir))
            return

        # Step 2: Load train split (Spider provides train.json and dev.json)
        logger.info("\n[2/5] Loading Spider training data...")
        train_examples = self.load_spider_split("train")

        # Step 3: Create train/val split from Spider's train.json
        # (We'll use Spider's official dev.json as test set later)
        logger.info("\n[3/5] Creating train/validation split...")
        train_subset, val_subset = self.create_train_val_split(train_examples)

        # Step 4: Load Spider's official dev set (for evaluation)
        logger.info("\n[4/5] Loading Spider dev set (for testing)...")
        dev_examples = self.load_spider_split("dev")

        # Step 5: Save formatted data
        logger.info("\n[5/5] Saving formatted datasets...")

        self.save_for_unsloth(
            train_subset,
            self.processed_dir / "train.json",
            include_schema=True
        )

        self.save_for_unsloth(
            val_subset,
            self.processed_dir / "validation.json",
            include_schema=True
        )

        self.save_for_unsloth(
            dev_examples,
            self.processed_dir / "test.json",
            include_schema=True
        )

        # Generate summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Training samples:   {len(train_subset):>6}")
        logger.info(f"Validation samples: {len(val_subset):>6}")
        logger.info(f"Test samples:       {len(dev_examples):>6}")
        logger.info(f"Total:              {len(train_subset) + len(val_subset) + len(dev_examples):>6}")
        logger.info("=" * 70)

        # Sample output
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLE FORMATTED EXAMPLE")
        logger.info("=" * 70)
        sample = self.format_as_alpaca_instruction(train_subset[0])
        logger.info(f"\nInstruction:\n{sample['instruction']}\n")
        logger.info(f"Input:\n{sample['input'][:300]}...\n")
        logger.info(f"Output:\n{sample['output']}\n")
        logger.info("=" * 70)

        logger.info("\n✓ Data preparation complete!")
        logger.info(f"Processed data saved to: {self.processed_dir}")


def main():
    """Main execution function."""
    # Set up paths
    base_dir = Path.home() / "nano-analyst" / "data"

    # Initialize processor
    processor = SpiderDatasetProcessor(base_dir)

    # Run pipeline
    processor.process_full_pipeline()


if __name__ == "__main__":
    main()
