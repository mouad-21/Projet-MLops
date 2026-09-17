"""API FastAPI qui sert le modele PROMU depuis le MLflow Model Registry."""
import sys
from pathlib import Path
from typing import List

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import load_config, get_tracking_uri  # noqa: E402

CONFIG_PATH = "configs/config.yaml"
cfg = load_config(CONFIG_PATH)
mlflow.set_tracking_uri(get_tracking_uri(cfg))

app = FastAPI(title="Credit Card Fraud Detection API")

_model = None


def get_model():
    global _model
    if _model is None:
        name = cfg.mlflow.registered_model_name
        try:
            uri = f"models:/{name}/{cfg.mlflow.promote_stage}"
            _model = mlflow.pyfunc.load_model(uri)
        except Exception:
            uri = f"models:/{name}@{cfg.mlflow.promote_stage.lower()}"
            _model = mlflow.pyfunc.load_model(uri)
    return _model


class Transaction(BaseModel):
    features: List[float]  # 29 valeurs, dans l'ordre config.features.numeric (V1..V28, Amount)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Transaction):
    if len(item.features) != len(cfg.features.numeric):
        return {"error": f"Expected {len(cfg.features.numeric)} features, got {len(item.features)}"}

    model = get_model()
    import pandas as pd
    df = pd.DataFrame([item.features], columns=cfg.features.numeric)
    pred = model.predict(df)[0]
    label = "fraud" if int(pred) == 1 else "legitimate"
    return {"prediction": int(pred), "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
