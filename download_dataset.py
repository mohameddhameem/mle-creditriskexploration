import os
import shutil

from datasets import load_dataset

import config


def download_dataset(dataset_id: str = "mohameddhameem/home-credit-default-risk", output_dir: str | None = None):
    """Download files from a HuggingFace dataset."""
    if output_dir is None:
        output_dir = config.DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    dataset_names = [
        'application_test',
        'application_train_dated',
        'bureau',
        'bureau_balance',
        'credit_card_balance',
        'installments_payments',
        'POS_CASH_balance',
        'previous_application',
        'sample_submission'
    ]

    for name in dataset_names:
        dataset = load_dataset('mohameddhameem/home-credit-default-risk', cache_dir=config.DATA_DIR, name=name)
        dataset['train'].to_parquet(os.path.join(config.DATA_DIR, f'{name}.parquet'))

    # Cleanup the cache directory
    cache_subdir = dataset_id.replace('/', '___')
    cleanup_path = os.path.join(config.DATA_DIR, cache_subdir)
    if os.path.exists(cleanup_path):
        shutil.rmtree(cleanup_path)
