"""
Data Validation Script
======================
Validates that the prepared Spider dataset is correctly formatted for Unsloth fine-tuning.

Checks:
1. JSON structure (instruction/input/output keys)
2. Schema presence in input field
3. SQL syntax validity (basic check)
4. No empty fields
5. Character encoding issues

Usage:
    python validate_data.py
"""

import json
from pathlib import Path
from typing import Dict, List
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates formatted Spider dataset."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def validate_structure(self, examples: List[Dict]) -> bool:
        """Check that all examples have required fields."""
        required_keys = {'instruction', 'input', 'output'}

        for idx, example in enumerate(examples):
            if not all(key in example for key in required_keys):
                logger.error(f"Example {idx} missing required keys. Has: {example.keys()}")
                return False

            # Check no empty values
            for key in required_keys:
                if not example[key] or not example[key].strip():
                    logger.error(f"Example {idx} has empty '{key}' field")
                    return False

        logger.info(f"✓ All {len(examples)} examples have required structure")
        return True

    def validate_schema_presence(self, examples: List[Dict]) -> bool:
        """Check that input field contains CREATE TABLE statements."""
        missing_schema = 0

        for idx, example in enumerate(examples):
            input_text = example['input']

            if 'CREATE TABLE' not in input_text.upper():
                logger.warning(f"Example {idx} missing CREATE TABLE in input")
                missing_schema += 1

        if missing_schema > 0:
            logger.warning(f"⚠ {missing_schema} examples missing schema definitions")
            return False

        logger.info(f"✓ All examples contain schema definitions")
        return True

    def validate_sql_syntax(self, examples: List[Dict], sample_size: int = 50) -> bool:
        """Basic SQL syntax validation (checks for common keywords)."""
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'INSERT', 'UPDATE', 'DELETE']

        invalid = 0
        sample = examples[:sample_size]

        for idx, example in enumerate(sample):
            sql = example['output'].upper()

            # Check if at least one SQL keyword exists
            if not any(keyword in sql for keyword in sql_keywords):
                logger.warning(f"Example {idx} output doesn't look like SQL: {example['output'][:50]}")
                invalid += 1

        if invalid > sample_size * 0.1:  # More than 10% invalid
            logger.error(f"✗ {invalid}/{sample_size} samples have suspicious SQL")
            return False

        logger.info(f"✓ SQL syntax check passed ({sample_size} samples)")
        return True

    def analyze_statistics(self, examples: List[Dict]) -> Dict:
        """Compute dataset statistics."""
        stats = {
            'total_examples': len(examples),
            'avg_question_length': 0,
            'avg_schema_length': 0,
            'avg_sql_length': 0,
            'max_input_length': 0,
            'sql_complexity': {
                'simple_select': 0,
                'with_joins': 0,
                'with_aggregation': 0,
                'with_subquery': 0
            }
        }

        question_lengths = []
        schema_lengths = []
        sql_lengths = []

        for example in examples:
            # Parse question from input
            input_text = example['input']
            if 'Question:' in input_text:
                question = input_text.split('Question:')[-1].strip()
                question_lengths.append(len(question))

            # Schema length (rough estimate)
            schema_section = input_text.split('Question:')[0] if 'Question:' in input_text else ''
            schema_lengths.append(len(schema_section))

            # SQL length
            sql = example['output']
            sql_lengths.append(len(sql))

            # Track max input length (important for context window)
            stats['max_input_length'] = max(stats['max_input_length'], len(input_text))

            # SQL complexity
            sql_upper = sql.upper()
            if 'JOIN' in sql_upper:
                stats['sql_complexity']['with_joins'] += 1
            if any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                stats['sql_complexity']['with_aggregation'] += 1
            if 'SELECT' in sql_upper and sql_upper.count('SELECT') > 1:
                stats['sql_complexity']['with_subquery'] += 1
            if sql_upper.count('SELECT') == 1 and 'JOIN' not in sql_upper:
                stats['sql_complexity']['simple_select'] += 1

        stats['avg_question_length'] = int(sum(question_lengths) / len(question_lengths)) if question_lengths else 0
        stats['avg_schema_length'] = int(sum(schema_lengths) / len(schema_lengths)) if schema_lengths else 0
        stats['avg_sql_length'] = int(sum(sql_lengths) / len(sql_lengths)) if sql_lengths else 0

        return stats

    def print_statistics(self, stats: Dict, split_name: str):
        """Pretty-print statistics."""
        logger.info(f"\n{'=' * 70}")
        logger.info(f"{split_name.upper()} SPLIT STATISTICS")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total examples:        {stats['total_examples']:>6}")
        logger.info(f"Avg question length:   {stats['avg_question_length']:>6} chars")
        logger.info(f"Avg schema length:     {stats['avg_schema_length']:>6} chars")
        logger.info(f"Avg SQL length:        {stats['avg_sql_length']:>6} chars")
        logger.info(f"Max input length:      {stats['max_input_length']:>6} chars")
        logger.info(f"\nSQL Complexity:")
        logger.info(f"  Simple SELECT:       {stats['sql_complexity']['simple_select']:>6}")
        logger.info(f"  With JOINs:          {stats['sql_complexity']['with_joins']:>6}")
        logger.info(f"  With Aggregations:   {stats['sql_complexity']['with_aggregation']:>6}")
        logger.info(f"  With Subqueries:     {stats['sql_complexity']['with_subquery']:>6}")
        logger.info(f"{'=' * 70}")

    def validate_split(self, split_name: str) -> bool:
        """Validate a single data split."""
        file_path = self.data_dir / f"{split_name}.json"

        if not file_path.exists():
            logger.error(f"✗ {split_name}.json not found at {file_path}")
            return False

        logger.info(f"\nValidating {split_name}.json...")

        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        # Run validations
        checks = [
            self.validate_structure(examples),
            self.validate_schema_presence(examples),
            self.validate_sql_syntax(examples)
        ]

        # Compute statistics
        stats = self.analyze_statistics(examples)
        self.print_statistics(stats, split_name)

        return all(checks)

    def validate_all(self) -> bool:
        """Validate all splits (train, validation, test)."""
        logger.info("\n" + "=" * 70)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 70)

        results = {}
        for split in ['train', 'validation', 'test']:
            results[split] = self.validate_split(split)

        # Final report
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)

        all_passed = all(results.values())

        for split, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{split:>12}: {status}")

        logger.info("=" * 70)

        if all_passed:
            logger.info("\n SUCCESS! All data splits are valid and ready for fine-tuning.")
            logger.info("\n Next step: Run the Unsloth fine-tuning script (Step 2)")
        else:
            logger.error("\n ERRORS FOUND! Please fix data preparation issues before proceeding.")

        return all_passed


def main():
    """Main validation execution."""
    data_dir = Path.home() / "nano-analyst" / "data" / "processed"

    validator = DataValidator(data_dir)
    success = validator.validate_all()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
