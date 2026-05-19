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
