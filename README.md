---
title: Home Credit Default Risk API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Home Credit Default Risk

A dataset for learning about credit default prediction.

## What is this?

This is a credit risk dataset from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk), available on [HuggingFace](https://huggingface.co/datasets/mohameddhameem/home-credit-default-risk).

It contains real-world loan application data with customer information, credit history, and whether they defaulted on their loan. Great for learning machine learning classification.

## The Data

- **Size**: ~1 million+ records
- **Type**: Tabular classification
- **License**: CC-BY-4.0
- **Main tables**: applications, credit history, payment history, etc.

## Quick Start

```python
from datasets import load_dataset

# Load the training data
dataset = load_dataset("mohameddhameem/home-credit-default-risk", "application_train")
```

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── data/                  (datasets - not tracked)
├── .gitignore            (ignores CSV, parquet, data/)
├── requirements.txt
└── README.md
```

## License & Attribution

Dataset from Kaggle competition. Used for educational purposes under CC-BY-4.0 license.

See the [Kaggle competition page](https://www.kaggle.com/competitions/home-credit-default-risk) for more details.

## Running the Inference API

Start the API:
```bash
docker compose up -d api
```

Query predictions:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"EXT_SOURCE_2": 0.65, "EXT_SOURCE_3": 0.52}}'
```

## Airflow & Model Monitoring Setup

An end-to-end model training, inference, and monitoring pipeline is orchestrated using Apache Airflow.

### 1. Starting the Airflow Stack
Start all Airflow services along with the Jupyter and API containers:
```bash
docker compose up --build -d
```
Access the Airflow Web UI at [http://localhost:8080](http://localhost:8080) (Credentials: `admin`/`admin`).

### 2. DAG Orchestration
The DAG `ml_credit_risk_pipeline` is scheduled **monthly** for the year 2020 (the Out-of-Time split window) with `catchup=True` to simulate a full year of production scoring and monitoring:
1. **`model_train`**: Runs model training on the historical 2018-2019 split.
2. **`model_inference`**: Scores applications booked during the current execution month and saves predictions to `datamart/predictions/YYYY-MM-DD/`.
3. **`model_monitor`**: Looks back 6 months to evaluate realized model performance (ROC-AUC) against ground-truth labels. It also calculates Population Stability Index (PSI) to detect drift in prediction distributions and top features (`EXT_SOURCE_2`, `EXT_SOURCE_3`, `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `cc_utilisation`).

### 3. Monitoring Results
Monitoring metrics are saved monthly as Parquet files under `datamart/gold/model_monitoring/monitor_YYYY_MM_DD.parquet`.

## CI Deployment (GHCR -> DigitalOcean)

Workflow file: `.github/workflows/deploy-do.yml`

Required GitHub repository secrets:
- `DIGITALOCEAN_ACCESS_TOKEN`
- `DO_APP_ID`

What happens on push to `main`:
- Builds and pushes API image to GHCR.
- Triggers a new DigitalOcean App Platform deployment via doctl.

