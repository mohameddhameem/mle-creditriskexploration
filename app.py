import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Dict, Any, List, Union, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE", "prev_last_status",
]


def get_latest_model_timestamp() -> str:
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Model directory {MODEL_DIR} does not exist.")

    metadata_files = glob.glob(os.path.join(MODEL_DIR, "metadata_*.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {MODEL_DIR}. Please run training first.")

    metadata_files.sort()
    latest_file = metadata_files[-1]
    filename = os.path.basename(latest_file)
    return filename.replace("metadata_", "").replace(".json", "")


try:
    latest_ts = get_latest_model_timestamp()
    meta_path = os.path.join(MODEL_DIR, f"metadata_{latest_ts}.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    feature_names = metadata["feature_names"]
    medians = metadata["medians"]
    logger.info(f"Loaded metadata for timestamp: {latest_ts}")
except Exception as e:
    logger.warning(f"Could not load metadata at startup: {e}.")
    latest_ts, feature_names, medians, metadata = None, [], {}, {}

encoders = {}
models = {}


def load_model_pickles():
    global encoders, models
    if not latest_ts:
        raise ValueError("No model timestamp detected. Train models first.")

    if models and encoders:
        return

    try:
        enc_path = os.path.join(MODEL_DIR, f"encoders_{latest_ts}.pkl")
        xgb_path = os.path.join(MODEL_DIR, f"xgboost_{latest_ts}.pkl")
        lr_path = os.path.join(MODEL_DIR, f"logistic_regression_{latest_ts}.pkl")

        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)
        with open(xgb_path, "rb") as f:
            models["xgboost"] = pickle.load(f)
        with open(lr_path, "rb") as f:
            models["logistic_regression"] = pickle.load(f)

        logger.info(f"Pickle artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load pickle artifacts: {e}", exc_info=True)
        raise e


if feature_names:
    fields = {}
    for name in feature_names:
        col_type = Optional[str] if name in CATEGORICAL_COLS else Optional[float]
        fields[name] = (col_type, None)
    FeaturesInput = create_model("FeaturesInput", **fields)
else:
    class FeaturesInput(BaseModel):
        class Config:
            extra = "allow"

app = FastAPI(
    title="Home Credit Default Risk Inference API",
    description="Online inference API for predicting loan default risk.",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    try:
        load_model_pickles()
    except Exception as e:
        logger.error(f"Startup loading failed: {e}")


def preprocess_input(data: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(data)

    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_names].copy()

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).fillna("missing")
        le = encoders[col]
        known = set(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else "missing")
        df[col] = le.transform(df[col])

    for col in df.columns:
        if col in CATEGORICAL_COLS:
            continue
        df[col] = df[col].fillna(medians.get(col, 0.0))

    return df.fillna(0.0)


class PredictRequest(BaseModel):
    inputs: Union[Dict[str, Any], List[Dict[str, Any]]]
    model_name: Optional[str] = "xgboost"


class PredictStructuredRequest(BaseModel):
    inputs: Union[FeaturesInput, List[FeaturesInput]]
    model_name: Optional[str] = "xgboost"


class PredictResponseItem(BaseModel):
    probability: float
    prediction: int


class PredictResponse(BaseModel):
    predictions: List[PredictResponseItem]
    model_used: str
    timestamp: str


@app.get("/")
def read_root():
    return {
        "message": "Home Credit Default Risk Inference Service is active.",
        "docs_url": "/docs",
        "latest_model_timestamp": latest_ts,
        "results": metadata.get("results", {})
    }


@app.get("/health")
def health():
    if not models or not encoders or not feature_names:
        return {
            "status": "degraded",
            "timestamp": latest_ts,
            "features_count": len(feature_names),
            "available_models": list(models.keys()),
            "detail": "Models or encoders are not loaded yet."
        }
    return {
        "status": "healthy",
        "timestamp": latest_ts,
        "features_count": len(feature_names),
        "available_models": list(models.keys())
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return run_inference(request.inputs, request.model_name)


@app.post("/predict_structured", response_model=PredictResponse)
def predict_structured(request: PredictStructuredRequest):
    raw_inputs = request.inputs
    if isinstance(raw_inputs, list):
        input_data = [item.dict(exclude_unset=True) for item in raw_inputs]
    else:
        input_data = [raw_inputs.dict(exclude_unset=True)]
    return run_inference(input_data, request.model_name)


def run_inference(inputs: Union[Dict[str, Any], List[Dict[str, Any]]], model_name: str = "xgboost") -> PredictResponse:
    load_model_pickles()
    model_name = model_name.lower()
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Choose from {list(models.keys())}."
        )

    raw_inputs = [inputs] if isinstance(inputs, dict) else inputs
    if not raw_inputs:
        raise HTTPException(status_code=400, detail="Inputs list cannot be empty.")

    try:
        processed_df = preprocess_input(raw_inputs)
        model = models[model_name]
        probabilities = model.predict_proba(processed_df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        response_items = [
            PredictResponseItem(probability=float(prob), prediction=int(pred))
            for prob, pred in zip(probabilities, predictions)
        ]

        return PredictResponse(
            predictions=response_items,
            model_used=model_name,
            timestamp=latest_ts
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
