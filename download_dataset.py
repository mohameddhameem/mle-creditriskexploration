import os
import shutil
import logging
from huggingface_hub import snapshot_download

import config

logger = logging.getLogger(__name__)


def download_dataset(dataset_id: str = "mohameddhameem/home-credit-default-risk", output_dir: str | None = None):
    """Download CSV/Parquet files from a HuggingFace dataset snapshot into DATA_DIR.

    If `output_dir` is None, `config.DATA_DIR` is used. Files whose names match the
    expected names in `config.RAW_FILES` (case-insensitive) will be renamed to the
    canonical names from config to ensure downstream ingestion finds them.
    """
    if output_dir is None:
        output_dir = config.DATA_DIR

    logger.info(f"Downloading files from '{dataset_id}' into '{output_dir}'...")

    # Download the repository snapshot into HF cache
    try:
        local_dir = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            ignore_patterns=["*.git*", "*.md"],
        )
    except Exception as exc:
        logger.error(f"HuggingFace snapshot_download failed: {exc}")
        raise

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Copying Parquet and CSV files from cache to '{output_dir}'...")

    # Build a case-insensitive mapping of expected filenames -> canonical name
    expected_map = {name.lower(): name for name in config.RAW_FILES.values()}

    files_copied = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if not file.endswith((".parquet", ".csv")):
                continue
            src_path = os.path.join(root, file)

            # If file name matches an expected name (case-insensitive), rename to canonical
            lower = file.lower()
            dest_name = expected_map.get(lower, file)
            dest_path = os.path.join(output_dir, dest_name)

            shutil.copy2(src_path, dest_path)
            logger.info(f"Copied {file} -> {dest_name}")
            files_copied += 1

    if files_copied > 0:
        logger.info(f"Success: {files_copied} files copied to '{output_dir}'.")
    else:
        logger.warning("No Parquet/CSV files found in the HuggingFace snapshot.")

    # Report any expected files that are still missing
    missing = [n for n in config.RAW_FILES.values() if not os.path.exists(os.path.join(output_dir, n))]
    if missing:
        logger.warning(f"The following expected files are missing in '{output_dir}': {missing}")


def main():
    download_dataset(output_dir=None)


if __name__ == "__main__":
    # Basic logging for standalone runs
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    main()
