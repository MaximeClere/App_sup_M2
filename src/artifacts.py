"""Enregistrement des résultats et métadonnées des exécutions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import shutil
from statistics import mean
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ExperimentConfig, new_run_name
from src.data import CompetitionData, TrainValidationSplit


def _sha256(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    """Récupère les versions des bibliothèques utilisées."""
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _class_distribution(values: pd.Series) -> dict[str, float]:
    """Calcule la proportion de chaque classe."""
    distribution = values.value_counts(normalize=True).sort_index()
    return {str(label): float(proportion) for label, proportion in distribution.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Écrit un objet JSON lisible dans un fichier."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def save_training_artifacts(
    config: ExperimentConfig,
    data: CompetitionData,
    split: TrainValidationSplit,
    model: Pipeline,
    probabilities: np.ndarray,
    score: float,
) -> Path:
    """Enregistre le modèle, les prédictions et les métriques de validation."""
    run_directory = config.artifacts_dir / new_run_name(config)
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.config_path, run_directory / "config.json")

    validation_predictions = pd.DataFrame(
        {
            config.id_column: split.validation_ids.to_numpy(),
            config.target_column: split.y_validation.to_numpy(),
            f"probability_{config.target_column}_1": probabilities,
        }
    )
    validation_predictions.to_csv(
        run_directory / "validation_predictions.csv",
        index=False,
        float_format="%.10g",
    )
    joblib.dump(model, run_directory / "validation_model.joblib")

    metadata = {
        "metric": config.metric,
        "validation_method": config.validation_method,
        "validation_score": score,
        "model": config.model_name,
        "model_params_configured": config.model_params,
        "model_params_resolved": model.named_steps["classifier"].get_params(deep=False),
        "master_seed": config.master_seed,
        "validation_size": config.validation_size,
        "preprocessing": {
            "method": config.preprocessing.method,
            "scale_numeric": config.preprocessing.scale_numeric,
            "add_missing_indicators": config.preprocessing.add_missing_indicators,
            "log1p_columns": list(config.preprocessing.log1p_columns),
            "missing_category_token": config.preprocessing.missing_category_token,
        },
        "train_rows": int(len(split.X_train)),
        "validation_rows": int(len(split.X_validation)),
        "train_class_distribution": _class_distribution(split.y_train),
        "validation_class_distribution": _class_distribution(split.y_validation),
        "feature_columns": list(data.features.columns),
        "train_csv_sha256": _sha256(config.data_dir / "train.csv"),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_directory / "metrics.json", metadata)
    return run_directory


def save_cross_validation_artifacts(
    config: ExperimentConfig,
    data: CompetitionData,
    model: Pipeline,
    fold_scores: list[float],
) -> Path:
    """Enregistre le ROC-AUC moyen et celui de chaque pli."""
    run_directory = config.artifacts_dir / new_run_name(config)
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.config_path, run_directory / "config.json")

    metadata = {
        "metric": config.metric,
        "validation_method": config.validation_method,
        "validation_score": mean(fold_scores),
        "fold_validation_scores": fold_scores,
        "n_splits": config.n_splits,
        "model": config.model_name,
        "model_params_configured": config.model_params,
        "model_params_resolved": model.named_steps["classifier"].get_params(deep=False),
        "master_seed": config.master_seed,
        "preprocessing": {
            "method": config.preprocessing.method,
            "scale_numeric": config.preprocessing.scale_numeric,
            "add_missing_indicators": config.preprocessing.add_missing_indicators,
            "log1p_columns": list(config.preprocessing.log1p_columns),
            "missing_category_token": config.preprocessing.missing_category_token,
        },
        "training_rows": int(len(data.features)),
        "class_distribution": _class_distribution(data.target),
        "feature_columns": list(data.features.columns),
        "train_csv_sha256": _sha256(config.data_dir / "train.csv"),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_directory / "metrics.json", metadata)
    return run_directory


def save_prediction_artifacts(
    config: ExperimentConfig,
    data: CompetitionData,
    model: Pipeline,
    submission: pd.DataFrame,
) -> tuple[Path, Path]:
    """Enregistre le modèle final, les métadonnées et le fichier de soumission."""
    run_directory = config.artifacts_dir / new_run_name(config)
    run_directory.mkdir(parents=True, exist_ok=False)
    config.submissions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config_path, run_directory / "config.json")

    model_path = run_directory / "final_model.joblib"
    joblib.dump(model, model_path)

    submission_path = config.submissions_dir / f"{run_directory.name}_submission.csv"
    submission.to_csv(submission_path, index=False, float_format="%.10g")

    metadata = {
        "model": config.model_name,
        "model_params_configured": config.model_params,
        "model_params_resolved": model.named_steps["classifier"].get_params(deep=False),
        "metric": config.metric,
        "master_seed": config.master_seed,
        "preprocessing": {
            "method": config.preprocessing.method,
            "scale_numeric": config.preprocessing.scale_numeric,
            "add_missing_indicators": config.preprocessing.add_missing_indicators,
            "log1p_columns": list(config.preprocessing.log1p_columns),
            "missing_category_token": config.preprocessing.missing_category_token,
        },
        "training_rows": int(len(data.features)),
        "test_rows": int(len(data.test_features)),
        "feature_columns": list(data.features.columns),
        "train_csv_sha256": _sha256(config.data_dir / "train.csv"),
        "test_csv_sha256": _sha256(config.data_dir / "test.csv"),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "submission_path": str(submission_path),
    }
    _write_json(run_directory / "prediction_metadata.json", metadata)
    return submission_path, model_path
