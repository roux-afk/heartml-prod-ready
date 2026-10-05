"""Model registry and sklearn pipeline construction."""

from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from heart_disease.config import RANDOM_STATE

ModelFactory = Callable[[], Any]

MODEL_REGISTRY: dict[str, ModelFactory] = {
    "svm": lambda: SVC(kernel="rbf", random_state=RANDOM_STATE),
    "naive_bayes": GaussianNB,
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "random_forest": lambda: RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE),
    "lightgbm": lambda: LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, verbose=-1),
    "xgboost": lambda: XGBClassifier(random_state=RANDOM_STATE),
}


def build_pipeline(model_name: str) -> Pipeline:
    """Create an unfitted pipeline: mean imputation -> scaling -> classifier."""
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", MODEL_REGISTRY[model_name]()),
        ]
    )
