"""Construction du pipeline scikit-learn : ColumnTransformer + modele."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric: list, categorical: list, model_type: str = "logreg") -> Pipeline:
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", num, numeric)]
    if categorical:
        transformers.append(("cat", cat, categorical))

    pre = ColumnTransformer(transformers=transformers)

    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(steps=[("pre", pre), ("model", model)])
