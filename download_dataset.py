import os
import shutil
from huggingface_hub import snapshot_download

def download_dataset(dataset_id="mohameddhameem/home-credit-default-risk", output_dir="data"):
    print(f"Downloading all files from '{dataset_id}'...")
    
    # Download the entire repository to a local cache
    # This is much faster and directly downloads the raw parquet files you uploaded
    local_dir = snapshot_download(
        repo_id=dataset_id, 
        repo_type="dataset",
        ignore_patterns=["*.git*", "*.md"] # Ignore git files and README
    )
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nMoving downloaded Parquet and CSV files to '{output_dir}/'...")
    
    # Find all data files in the downloaded cache and copy them to the output dir
    files_copied = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith((".parquet", ".csv")):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(output_dir, file)
                
                # Only copy if the file doesn't already exist or we want to overwrite
                shutil.copy2(src_path, dest_path)
                print(f" -> Copied {file}")
                files_copied += 1
                
    if files_copied > 0:
        print(f"\nSuccess! {files_copied} files downloaded to '{output_dir}/'.")
    else:
        print("\nNo data files found in the repository.")

def main():
    download_dataset(output_dir="parquet_files")

if __name__ == "__main__":
    main()
