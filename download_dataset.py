import os
import shutil
from huggingface_hub import snapshot_download
import config

def download_dataset(dataset_id: str = "mohameddhameem/home-credit-default-risk", output_dir: str | None = None):
    """Download files from a HuggingFace dataset."""
    if output_dir is None:
        output_dir = config.DATA_DIR

    try:
        local_dir = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            ignore_patterns=["*.git*", "*.md"],
        )
    except Exception as exc:
        raise

    os.makedirs(output_dir, exist_ok=True)

    expected_map = {name.lower(): name for name in config.RAW_FILES.values()}

    files_copied = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if not file.endswith((".parquet", ".csv")):
                continue
            src_path = os.path.join(root, file)

            lower = file.lower()
            dest_name = expected_map.get(lower, file)
            dest_path = os.path.join(output_dir, dest_name)

            shutil.copy2(src_path, dest_path)
            files_copied += 1

