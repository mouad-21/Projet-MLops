"""Chargement du CSV brut + split train/test stratifie (la target est tres desequilibree)."""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config


def load_raw(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.data.raw_path)


def split_data(df: pd.DataFrame, cfg):
    X = df.drop(columns=[cfg.data.target])
    y = df[cfg.data.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
        stratify=y,  # essentiel vu le desequilibre extreme (0.17% de fraude)
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_raw(config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Fraude train: {y_train.mean():.3%} | Fraude test: {y_test.mean():.3%}")
