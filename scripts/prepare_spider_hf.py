"""
Alternative Spider Data Preparation Using HuggingFace Datasets
===============================================================
Simplified version that downloads Spider from HuggingFace Hub instead of Google Drive.
Faster and more reliable for most users.

Usage:
    pip install datasets
    python prepare_spider_hf.py
"""

import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_spider_to_alpaca(example: Dict) -> Dict[str, str]:
    """
    Convert HuggingFace Spider format to Alpaca instruction format.

    Args:
        example: Dictionary with 'question', 'query', and 'db_id' keys

    Returns:
        Formatted dictionary with instruction/input/output
    """
    # Extract schema from the example (HF Spider includes schema info)
    schema_parts = []

    # HuggingFace Spider provides table and column information
    if 'db_table_names' in example and 'db_column_names' in example:
        table_names = example['db_table_names']
        column_info = example['db_column_names']

        # Group columns by table
        from collections import defaultdict
        table_cols = defaultdict(list)

        for col_data in column_info:
            table_idx = col_data['table_id']
            col_name = col_data['column_name']

            if table_idx >= 0 and table_idx < len(table_names):
                table_cols[table_idx].append(col_name)

        # Generate CREATE TABLE statements
        for table_idx, table_name in enumerate(table_names):
            cols = table_cols.get(table_idx, [])
            if cols:
                cols_formatted = ",\n  ".join([f"{c} TEXT" for c in cols])
                schema_parts.append(
                    f"CREATE TABLE {table_name} (\n  {cols_formatted}\n);"
                )

    schema = "\n\n".join(schema_parts) if schema_parts else "-- Schema not available --"

    instruction = (
        "You are an expert SQL generator. Given a database schema and a natural language question, "
        "generate the correct SQL query to answer the question. "
        "Output only the SQL query without any explanations."
    )

    input_text = (
        f"Database Schema:\n{schema}\n\n"
        f"Question: {example['question']}"
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": example['query']
    }


def main():
    """Main execution with HuggingFace Datasets."""
    logger.info("=" * 70)
    logger.info("NANO-ANALYST DATA PREP (HuggingFace Method)")
    logger.info("=" * 70)

    # Setup paths
    base_dir = Path.home() / "nano-analyst" / "data" / "processed"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load Spider from HuggingFace
    logger.info("\n[1/3] Downloading Spider from HuggingFace Hub...")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not found. Install with: pip install datasets")
        return

    spider = load_dataset("spider")

    logger.info(f"✓ Downloaded Spider dataset")
    logger.info(f"  Train examples: {len(spider['train'])}")
    logger.info(f"  Validation examples: {len(spider['validation'])}")

    # Process train split
    logger.info("\n[2/3] Formatting train split...")
    train_formatted = []

    for example in spider['train']:
        try:
            formatted = format_spider_to_alpaca(example)
            train_formatted.append(formatted)
        except Exception as e:
            logger.warning(f"Skipped example due to error: {e}")

    # Create train/val split from Spider's train set
    import random
    random.seed(42)
    random.shuffle(train_formatted)

    split_idx = int(len(train_formatted) * 0.9)
    train_subset = train_formatted[:split_idx]
    val_subset = train_formatted[split_idx:]

    # Process test split (Spider's validation = our test set)
    logger.info("\n[3/3] Formatting test split...")
    test_formatted = []

    for example in spider['validation']:
        try:
            formatted = format_spider_to_alpaca(example)
            test_formatted.append(formatted)
        except Exception as e:
            logger.warning(f"Skipped example due to error: {e}")

    # Save files
    logger.info("\nSaving formatted datasets...")

    with open(base_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_subset, f, indent=2, ensure_ascii=False)

    with open(base_dir / "validation.json", 'w', encoding='utf-8') as f:
        json.dump(val_subset, f, indent=2, ensure_ascii=False)

    with open(base_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_formatted, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Training samples:   {len(train_subset):>6}")
    logger.info(f"Validation samples: {len(val_subset):>6}")
    logger.info(f"Test samples:       {len(test_formatted):>6}")
    logger.info(f"Total:              {len(train_subset) + len(val_subset) + len(test_formatted):>6}")
    logger.info("=" * 70)

    # Show sample
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE FORMATTED EXAMPLE")
    logger.info("=" * 70)
    sample = train_subset[0]
    logger.info(f"\nInstruction: {sample['instruction'][:100]}...")
    logger.info(f"\nInput:\n{sample['input'][:400]}...")
    logger.info(f"\nOutput:\n{sample['output']}")
    logger.info("=" * 70)

    logger.info(f"\n✓ Data preparation complete!")
    logger.info(f"Files saved to: {base_dir}")
    logger.info("\n Next Step: Run the fine-tuning script with Unsloth!")


if __name__ == "__main__":
    main()
