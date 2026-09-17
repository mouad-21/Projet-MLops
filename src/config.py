"""Chargement de la configuration YAML en objet Python typé."""
import os
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    openml_id: int
    raw_path: str
    target: str
    test_size: float
    random_state: int


@dataclass
class FeaturesConfig:
    numeric: list
    categorical: list


@dataclass
class ModelConfig:
    type: str
    params: dict


@dataclass
class CVConfig:
    strategy: str
    n_splits: int
    scoring: str


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    promote_stage: str


@dataclass
class Config:
    data: DataConfig
    features: FeaturesConfig
    model: ModelConfig
    cv: CVConfig
    mlflow: MLflowConfig


def load_config(path: str = "configs/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        model=ModelConfig(**raw["model"]),
        cv=CVConfig(**raw["cv"]),
        mlflow=MLflowConfig(**raw["mlflow"]),
    )


def get_tracking_uri(cfg: Config) -> str:
    """MLFLOW_TRACKING_URI (variable d'environnement) prend le pas sur la config.
    Utile pour le conteneur Docker, qui doit joindre le serveur MLflow via
    'http://host.docker.internal:5000' plutot que 'http://127.0.0.1:5000'."""
    return os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
