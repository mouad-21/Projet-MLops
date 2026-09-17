"""Entraine le pipeline avec GridSearchCV, tracke tout dans MLflow (autolog),
et enregistre le meilleur modele dans le Model Registry."""
import argparse

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import load_config
from pipeline import build_pipeline
from preprocess import load_raw, split_data


def main(cfg) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.sklearn.autolog(log_models=False, max_tuning_runs=5)  # on logge le modele nous-memes (voir plus bas)

    df = load_raw(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    pipe = build_pipeline(cfg.features.numeric, cfg.features.categorical, cfg.model.type)

    raw_grid = cfg.model.params[cfg.model.type]
    param_grid = {f"model__{k}": v for k, v in raw_grid.items()}

    cv = StratifiedKFold(n_splits=cfg.cv.n_splits, shuffle=True, random_state=cfg.data.random_state)

    search = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=cfg.cv.scoring,
        n_jobs=-1, refit=True, verbose=1,
    )

    with mlflow.start_run(run_name=f"train-{cfg.model.type}") as run:
        print(f"Lancement de GridSearchCV ({cfg.model.type}), scoring={cfg.cv.scoring}...")
        search.fit(X_train, y_train)

        print(f"Best params: {search.best_params_}")
        print(f"Best CV {cfg.cv.scoring}: {search.best_score_:.4f}")

        # Le meilleur pipeline (feature engineering + modele) est logge ET enregistre
        # dans le Model Registry en un seul appel.
        # skops_trusted_types : le modele est le notre (on vient de l'entrainer),
        # on peut donc explicitement faire confiance aux types internes de scikit-learn
        # (RandomForest/DecisionTree) que skops signale par securite par defaut.
        mlflow.sklearn.log_model(
            search.best_estimator_,
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )

        client = MlflowClient()
        latest = client.get_latest_versions(cfg.mlflow.registered_model_name)
        new_version = max(int(v.version) for v in latest)

        try:
            # API "stages" (MLflow < ~2.9)
            client.transition_model_version_stage(
                name=cfg.mlflow.registered_model_name,
                version=new_version,
                stage=cfg.mlflow.promote_stage,
                archive_existing_versions=True,
            )
            print(f"Version {new_version} promue au stage '{cfg.mlflow.promote_stage}'.")
        except Exception as e:
            print(f"API 'stages' indisponible ou depreciee ({e}); bascule sur les alias.")
            # API "aliases" (MLflow recent) : on retombe dessus si "stages" n'existe plus
            alias = cfg.mlflow.promote_stage.lower()
            client.set_registered_model_alias(cfg.mlflow.registered_model_name, alias, new_version)
            print(f"Version {new_version} taguee avec l'alias '{alias}'.")

        print(f"Run ID: {run.info.run_id}")
        print(f"Modele enregistre : {cfg.mlflow.registered_model_name} (version {new_version})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
