"""
Download Spider Databases
==========================
Downloads the actual SQLite databases from Spider dataset.
Required for evaluation to execute queries.
"""

import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_spider_databases():
    """Download Spider databases from official source."""

    print("=" * 70)
    print("DOWNLOADING SPIDER DATABASES")
    print("=" * 70)

    # Paths
    project_root = Path.home() / "nano-analyst"
    data_dir = project_root / "data"
    spider_dir = data_dir / "spider"
    spider_dir.mkdir(parents=True, exist_ok=True)

    # Download URLs (from Spider official GitHub)
    database_url = "https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"

    print("\n[1/3] Downloading Spider databases (220MB)...")
    print("This may take a few minutes...")

    zip_path = spider_dir / "spider_databases.zip"

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(
            database_url,
            zip_path,
            reporthook=report_progress
        )
        print("\n✓ Download complete!")

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Manual download required")
        print("1. Visit: https://yale-lily.github.io/spider")
        print("2. Download 'Spider 1.0 SQL and Databases'")
        print(f"3. Extract to: {spider_dir}")
        return False

    # Extract
    print("\n[2/3] Extracting databases...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(spider_dir)
        print("✓ Extraction complete!")

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

    # Verify
    print("\n[3/3] Verifying databases...")

    database_dir = spider_dir / "database"
    if database_dir.exists():
        db_count = len(list(database_dir.glob("*/*.sqlite")))
        print(f"✓ Found {db_count} databases")

        # List some examples
        print("\nSample databases:")
        for db_path in list(database_dir.glob("*/*.sqlite"))[:5]:
            print(f"  - {db_path.parent.name}/{db_path.name}")

        print(f"\n✅ Setup complete!")
        print(f"Databases location: {database_dir}")

        # Clean up zip
        zip_path.unlink()
        print(f"✓ Cleaned up temporary files")

        return True
    else:
        print("✗ Database directory not found after extraction")
        return False


if __name__ == "__main__":
    success = download_spider_databases()

    if success:
        print("\n" + "=" * 70)
        print("READY FOR EVALUATION!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run: python evaluate_agent.py --test-data data/processed/test.json --databases-dir data/spider/database --max-examples 10")
    else:
        print("\nPlease download databases manually from: https://yale-lily.github.io/spider")
