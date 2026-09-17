"""Recupere le dataset Credit Card Fraud Detection (OpenML) et le sauvegarde en local."""
import argparse
from pathlib import Path

from sklearn.datasets import fetch_openml

from config import load_config


def get_data(cfg) -> None:
    print(f"Telechargement du dataset OpenML id={cfg.data.openml_id}...")
    data = fetch_openml(data_id=cfg.data.openml_id, as_frame=True, parser="auto")
    df = data.frame

    # La cible arrive parfois en categorie '0'/'1' (string) : on la force en int.
    df[cfg.data.target] = df[cfg.data.target].astype(int)

    out_path = Path(cfg.data.raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    n_fraud = int(df[cfg.data.target].sum())
    print(f"Sauvegarde dans {out_path} : {len(df)} lignes, {n_fraud} fraudes ({n_fraud / len(df):.3%}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    get_data(config)
