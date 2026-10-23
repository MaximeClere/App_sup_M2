"""Fit one configured model and evaluate it on a stratified holdout set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import ExperimentConfig, load_config, new_run_name
from src.data import load_competition_data, split_train_validation
from src.models import build_model_pipeline, positive_class_probabilities


def _seed_global_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _class_distribution(values: pd.Series) -> dict[str, float]:
    distribution = values.value_counts(normalize=True).sort_index()
    return {str(label): float(proportion) for label, proportion in distribution.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run_training(config: ExperimentConfig) -> Path:
    """Train a configured estimator and save validation artifacts."""
    _seed_global_generators(config.master_seed)
    data = load_competition_data(config)
    split = split_train_validation(
        data=data,
        validation_size=config.validation_size,
        master_seed=config.master_seed,
    )

    model = build_model_pipeline(config, data.features.columns)
    model.fit(split.X_train, split.y_train)
    probabilities = positive_class_probabilities(model, split.X_validation)
    score = float(roc_auc_score(split.y_validation, probabilities))

    run_name = new_run_name(config)
    run_directory = config.artifacts_dir / run_name
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

    train_path = config.data_dir / "train.csv"
    metadata = {
        "metric": config.metric,
        "validation_score": score,
        "model": config.model_name,
        "model_params_configured": config.model_params,
        "model_params_resolved": model.named_steps["classifier"].get_params(deep=False),
        "master_seed": config.master_seed,
        "validation_size": config.validation_size,
        "preprocessing": {
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
        "train_csv_sha256": _sha256(train_path),
        "python_version": sys.version.split()[0],
        "package_versions": _package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_directory / "metrics.json", metadata)

    print(f"Modèle : {config.model_name}")
    print(f"ROC-AUC validation : {score:.6f}")
    print(f"Résultats enregistrés dans : {run_directory}")
    return run_directory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle configuré et mesure son ROC-AUC de validation."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Chemin vers le fichier de configuration JSON (défaut : config.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
