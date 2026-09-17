"""Charge le modele PROMU depuis le MLflow Model Registry (pas depuis la memoire !)
et l'evalue sur le jeu de test tenu a l'ecart. Logge metriques + plots dans un nouveau run."""
import argparse
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # evite un crash sur les logs MLflow contenant des emojis (Windows/cp1252)

import mlflow
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import load_config, get_tracking_uri
from preprocess import load_raw, split_data
from utils import plot_confusion_matrix, plot_pr_curve, plot_roc_curve


def load_production_model(cfg):
    """Essaie d'abord l'API 'stages', puis l'API 'aliases' si besoin (selon version MLflow)."""
    name = cfg.mlflow.registered_model_name
    try:
        uri = f"models:/{name}/{cfg.mlflow.promote_stage}"
        return mlflow.pyfunc.load_model(uri), uri
    except Exception:
        uri = f"models:/{name}@{cfg.mlflow.promote_stage.lower()}"
        return mlflow.pyfunc.load_model(uri), uri


def main(cfg) -> None:
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    df = load_raw(cfg)
    _, X_test, _, y_test = split_data(df, cfg)  # meme split que train.py (random_state fixe)

    model, model_uri = load_production_model(cfg)
    print(f"Modele charge depuis le registry : {model_uri}")

    y_pred = model.predict(X_test)
    # mlflow.pyfunc renvoie parfois des probas/labels selon le flavor ; on normalise en 0/1
    y_pred_binary = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred

    metrics = {
        "test_precision": precision_score(y_test, y_pred_binary),
        "test_recall": recall_score(y_test, y_pred_binary),
        "test_f1": f1_score(y_test, y_pred_binary),
        "test_roc_auc": roc_auc_score(y_test, y_pred_binary),
        "test_average_precision": average_precision_score(y_test, y_pred_binary),
    }

    print("\nClassification report (test set) :")
    print(classification_report(y_test, y_pred_binary, target_names=["Legitimate", "Fraud"]))
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with mlflow.start_run(run_name="evaluate-production-model"):
        mlflow.log_params({"evaluated_model_uri": model_uri})
        mlflow.log_metrics(metrics)

        cm_path = plot_confusion_matrix(y_test, y_pred_binary, "artifacts/confusion_matrix.png")
        roc_path = plot_roc_curve(y_test, y_pred_binary, "artifacts/roc_curve.png")
        pr_path = plot_pr_curve(y_test, y_pred_binary, "artifacts/pr_curve.png")

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)

    print("\nEvaluation loggee dans MLflow (run 'evaluate-production-model').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
