"""
Download Full Spider Dataset with Databases
============================================
Downloads Spider with databases from HuggingFace and sets up
evaluation environment.
"""

import json
from pathlib import Path
from datasets import load_dataset


def download_and_prepare():
    """Download Spider with all metadata for evaluation."""

    print("=" * 70)
    print("DOWNLOADING FULL SPIDER DATASET")
    print("=" * 70)

    # Paths
    project_root = Path.home() / "nano-analyst"
    data_dir = project_root / "data"
    eval_dir = data_dir / "spider_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Loading Spider from HuggingFace (with database metadata)...")

    # Load with database info
    dataset = load_dataset("spider", trust_remote_code=True)

    print(f"✓ Loaded Spider dataset")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Validation: {len(dataset['validation'])} examples")

    # Save validation set with full metadata for evaluation
    print("\n[2/2] Preparing evaluation data...")

    validation_data = []

    for example in dataset['validation']:
        validation_data.append({
            "db_id": example['db_id'],
            "question": example['question'],
            "query": example['query'],
            "query_toks": example.get('query_toks', []),
            "query_toks_no_value": example.get('query_toks_no_value', []),
            "question_toks": example.get('question_toks', [])
        })

    # Save
    eval_data_path = eval_dir / "validation_eval.json"
    with open(eval_data_path, 'w') as f:
        json.dump(validation_data, f, indent=2)

    print(f"✓ Saved evaluation data: {eval_data_path}")
    print(f"  Total examples: {len(validation_data)}")

    # Show unique databases
    unique_dbs = set(ex['db_id'] for ex in validation_data)
    print(f"  Unique databases: {len(unique_dbs)}")

    # Show sample
    print(f"\n📋 Sample evaluation example:")
    sample = validation_data[0]
    print(f"  Database: {sample['db_id']}")
    print(f"  Question: {sample['question']}")
    print(f"  Gold SQL: {sample['query']}")

    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE")
    print("=" * 70)

    return eval_data_path, unique_dbs


def download_spider_databases_manual():
    """Instructions for downloading databases."""

    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOAD DATABASES")
    print("=" * 70)
    print("\nThe HuggingFace dataset doesn't include the actual .sqlite files.")
    print("You need to download them separately.\n")
    print("Option 1: Direct download (recommended)")
    print("-" * 70)
    print("Run these commands:")
    print()
    print("  cd ~/nano-analyst/data")
    print("  wget https://github.com/taoyds/spider/raw/master/database.zip")
    print("  unzip database.zip")
    print()
    print("Option 2: Manual download")
    print("-" * 70)
    print("1. Visit: https://github.com/taoyds/spider")
    print("2. Download 'database.zip'")
    print("3. Extract to: ~/nano-analyst/data/database/")
    print()
    print("=" * 70)


if __name__ == "__main__":
    eval_data_path, unique_dbs = download_and_prepare()

    print(f"\n✓ Evaluation data ready at: {eval_data_path}")
    print(f"\nDatabases needed ({len(unique_dbs)} total):")
    for db in sorted(unique_dbs)[:10]:
        print(f"  - {db}")
    if len(unique_dbs) > 10:
        print(f"  ... and {len(unique_dbs) - 10} more")

    # Show download instructions
    download_spider_databases_manual()
