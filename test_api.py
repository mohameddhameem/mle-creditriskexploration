import glob
import numpy as np
import os
import pandas as pd
import requests
import time

# URL of the API inside the Docker network (running as service 'api')
# If running this script inside the container, use 'http://api:8000'
API_URL = "http://api:8000"


def wait_for_api(url, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("API is up and healthy!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        print("Waiting for API to start...")
        time.sleep(2)
    print("Timeout waiting for API.")
    return False


def test_api():
    print(f"Connecting to API at {API_URL}...")
    if not wait_for_api(API_URL):
        exit(1)

    # 1. Health check
    res = requests.get(f"{API_URL}/health")
    print(f"Health Check Response: {res.json()}")

    # 2. Get a sample row from the gold test split to use as request payload
    gold_test_dir = "/app/datamart/gold/feature_store/test"
    parquet_files = glob.glob(os.path.join(gold_test_dir, "*.parquet"))

    if not parquet_files:
        # If no parquet found in gold (due to partitioning, let's search recursively)
        parquet_files = glob.glob(os.path.join(gold_test_dir, "**", "*.parquet"), recursive=True)

    if not parquet_files:
        print("No test parquet files found, using synthetic test data.")
        # Fallback synthetic sample
        sample_input = {
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 0.0,
            "AMT_INCOME_TOTAL": 135000.0,
            "EXT_SOURCE_2": 0.65,
            "EXT_SOURCE_3": 0.52
        }
    else:
        # Load the first parquet file and extract the first row as a dictionary
        print(f"Loading sample from: {parquet_files[0]}")
        df = pd.read_parquet(parquet_files[0])
        # Drop SK_ID_CURR to match expected features
        if "SK_ID_CURR" in df.columns:
            df = df.drop(columns=["SK_ID_CURR"])
        # Take the first row as a dictionary
        sample_input = df.iloc[0].to_dict()

        # Convert any float types that pandas/numpy might have wrapped
        # JSON serializer doesn't like np.float32, np.int64, etc.
        for k, v in sample_input.items():
            if pd.isna(v):
                sample_input[k] = None
            elif isinstance(v, (int, np.integer)):
                sample_input[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                sample_input[k] = float(v)

    # 3. Test freeform POST /predict (XGBoost)
    print("\nTesting freeform prediction (XGBoost)...")
    payload = {
        "inputs": sample_input,
        "model_name": "xgboost"
    }

    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error Response: {response.text}")

    # 4. Test freeform POST /predict (Logistic Regression)
    print("\nTesting freeform prediction (Logistic Regression)...")
    payload["model_name"] = "logistic_regression"
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error Response: {response.text}")

    # 5. Test structured POST /predict_structured
    print("\nTesting structured prediction (XGBoost)...")
    payload_structured = {
        "inputs": [sample_input],
        "model_name": "xgboost"
    }
    response = requests.post(f"{API_URL}/predict_structured", json=payload_structured)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error Response: {response.text}")

    print("\nAll integration tests complete.")


if __name__ == "__main__":
    test_api()
