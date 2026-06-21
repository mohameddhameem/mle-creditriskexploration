import argparse
import sys
import os

# Add repo root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import main as run_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshotdate", required=True)
    args = parser.parse_args()
    
    print(f"Starting model training for snapshot date: {args.snapshotdate}")
    run_train()
    print("Model training complete.")

if __name__ == "__main__":
    main()
