"""Registre des modèles scikit-learn et outils de prédiction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import ExperimentConfig
from src.preprocessing import build_preprocessor


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "xgboost": XGBClassifier,
}

DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 2000,
        "solver": "liblinear",
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": 1,
    },
    "extra_trees": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": 1,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": 1,
    },
}


def build_model_pipeline(
    config: ExperimentConfig,
    feature_columns: Sequence[str],
) -> Pipeline:
    """Assemble le prétraitement et le modèle dans une pipeline scikit-learn."""
    estimator_class = MODEL_REGISTRY[config.model_name]
    params = DEFAULT_MODEL_PARAMS[config.model_name].copy()
    params.update(config.model_params)

    try:
        estimator = estimator_class(
            **params,
            random_state=config.master_seed,
        )
    except TypeError as exc:
        raise ValueError(
            f"Hyperparamètres invalides pour {config.model_name}: {exc}"
        ) from exc

    preprocessor = build_preprocessor(feature_columns, config.preprocessing)
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", estimator),
        ]
    )


def positive_class_probabilities(model: Pipeline, X: Any) -> np.ndarray:
    """Retourne les probabilités de la classe positive 1."""
    classifier = model.named_steps["classifier"]
    classes = np.asarray(classifier.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError(
            f"Le modèle doit apprendre la classe positive 1; classes trouvées : {classes}."
        )
    return model.predict_proba(X)[:, int(positive_indices[0])]
