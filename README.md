# Projet MLOps — Détection de fraude par carte bancaire

Projet final du cours MLOps (M2 Campus Cyber, Ali Mokh). Pipeline complet de bout
en bout : récupération des données → prétraitement → entraînement avec tuning
d'hyperparamètres → tracking et registry **MLflow** → API de service (FastAPI) →
conteneurisation (Docker). Tout tourne **en local**, aucun service cloud payant.

## Dataset

[Credit Card Fraud Detection](https://www.openml.org/search?type=data&id=1597)
(OpenML id `1597`, correspond au dataset Kaggle `mlg-ulb/creditcardfraud`) :
284 807 transactions, **492 fraudes (0.17%)** — dataset volontairement très
déséquilibré, cas d'usage MLOps réaliste (nécessite un suivi/monitoring
particulier en production).

Toutes les features sont numériques : `V1`...`V28` (composantes issues d'une
PCA, pour anonymiser les données originales) + `Amount`. Cible : `Class`
(0 = légitime, 1 = fraude).

## Structure du projet

```
.
├─ data/                    # donnees brutes (gitignored, regenerees via `make get-data`)
├─ configs/
│  └─ config.yaml           # chemins, features, hyperparametres, config MLflow — rien en dur dans le code
├─ src/
│  ├─ config.py             # charge config.yaml en objet Python type
│  ├─ get_data.py           # telecharge le dataset (OpenML) -> data/raw.csv
│  ├─ preprocess.py         # split train/test stratifie (dataset tres desequilibre)
│  ├─ pipeline.py           # ColumnTransformer + modele (scikit-learn Pipeline)
│  ├─ train.py              # GridSearchCV + autolog MLflow + enregistrement dans le Model Registry
│  ├─ evaluate.py           # charge le modele PROMU depuis le registry, evalue sur le test set
│  └─ utils.py              # plots (matrice de confusion, ROC, precision-recall)
├─ tests/
│  └─ test_pipeline.py      # tests unitaires (pytest)
├─ app.py                   # API FastAPI, sert le modele depuis le MLflow Model Registry
├─ Makefile
├─ requirements.txt
├─ Dockerfile
└─ configs/config.yaml
```

## Pourquoi MLflow, concrètement

- **Tracking** : `mlflow.sklearn.autolog()` est actif pendant `train.py` — chaque
  combinaison d'hyperparamètres testée par `GridSearchCV` est trackée
  (paramètres, métriques de validation croisée).
- **Registry** : le meilleur pipeline (feature engineering + modèle) est
  enregistré sous le nom `CreditCardFraudClassifier`, puis **promu** au stage
  `Production` (ou tagué avec l'alias `production` selon la version de MLflow
  installée — le code gère les deux).
- **`evaluate.py` charge le modèle depuis le registry** (pas depuis la mémoire
  du script d'entraînement) — ça prouve que le cycle complet
  entraînement → enregistrement → chargement en aval fonctionne réellement,
  comme en production.
- **`app.py` fait pareil** : l'API ne connaît jamais le pipeline scikit-learn
  directement, elle le charge via `models:/CreditCardFraudClassifier/Production`.

## Choix techniques notables

- **Scoring = `average_precision`** (pas `accuracy`) pour le tuning : avec
  0.17% de fraude, un modèle qui prédit toujours "légitime" aurait 99.8%
  d'accuracy tout en étant inutile. `average_precision` (aire sous la courbe
  precision-recall) est bien plus pertinent sur un problème aussi déséquilibré.
- **`class_weight="balanced"`** sur les deux modèles testés (Logistic
  Regression, Random Forest) pour compenser le déséquilibre sans sur-échantillonner.
- **Split stratifié** (`stratify=y`) pour garder la même proportion de fraude
  dans train et test.
- **MLflow backend SQLite** (`sqlite:///mlflow.db`) : tracking + registry
  complets, 100% local, aucun serveur à lancer.

## Installation et exécution

```bash
make init                 # cree .venv et installe requirements.txt
source .venv/bin/activate # (ou .venv\Scripts\activate sous Windows)

make get-data              # telecharge le dataset dans data/raw.csv
make train                 # GridSearchCV + tracking MLflow + enregistrement du meilleur modele
make evaluate               # charge le modele "Production" depuis le registry, evalue sur le test set
make test                   # tests unitaires (pytest)

make mlflow-ui               # lance l'UI MLflow sur http://localhost:5000
make run                      # lance l'API FastAPI sur http://localhost:8000
make build                     # construit l'image Docker
```

### Tester l'API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [<29 valeurs : V1..V28, Amount>]}'
```

Réponse : `{"prediction": 0, "label": "legitimate"}` ou `{"prediction": 1, "label": "fraud"}`.

## Résultats

**Meilleur modèle (GridSearchCV, 4 candidats x 5 folds) :** Random Forest
(`n_estimators=200`, `max_depth=20`, `class_weight="balanced"`)
— average_precision (CV) = **0.843**

**Évaluation sur le jeu de test tenu à l'écart** (modèle rechargé depuis le
Model Registry, stage `Production`) :

| Métrique | Valeur |
|---|---|
| Precision (fraude) | 0.886 |
| Recall (fraude) | 0.796 |
| F1-score (fraude) | 0.839 |
| ROC AUC | 0.898 |
| Average Precision | 0.706 |

Voir `artifacts/` pour la matrice de confusion, la courbe ROC et la courbe
precision-recall, et l'UI MLflow (`make mlflow-ui`) pour comparer tous les
runs en détail (tracking + registry).
