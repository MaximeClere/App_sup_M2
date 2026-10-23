"""Fit the selected model on all labeled rows and write a Kaggle submission."""

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

from src.config import ExperimentConfig, load_config, new_run_name
from src.data import build_submission, load_competition_data
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def run_prediction(config: ExperimentConfig) -> Path:
    """Retrain on all of train.csv and save the final model and submission."""
    _seed_global_generators(config.master_seed)
    data = load_competition_data(config)

    model = build_model_pipeline(config, data.features.columns)
    model.fit(data.features, data.target)
    probabilities = positive_class_probabilities(model, data.test_features)
    submission = build_submission(
        template=data.submission_template,
        test_ids=data.test_ids,
        probabilities=probabilities,
        id_column=config.id_column,
        target_column=config.target_column,
    )

    run_name = new_run_name(config)
    run_directory = config.artifacts_dir / run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    config.submissions_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.config_path, run_directory / "config.json")

    model_path = run_directory / "final_model.joblib"
    joblib.dump(model, model_path)

    submission_path = config.submissions_dir / f"{run_name}_submission.csv"
    submission.to_csv(submission_path, index=False, float_format="%.10g")

    metadata = {
        "model": config.model_name,
        "model_params_configured": config.model_params,
        "model_params_resolved": model.named_steps["classifier"].get_params(deep=False),
        "metric": config.metric,
        "master_seed": config.master_seed,
        "preprocessing": {
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

    print(f"Modèle réentraîné sur {len(data.features)} lignes.")
    print(f"Soumission enregistrée dans : {submission_path}")
    print(f"Modèle enregistré dans : {model_path}")
    return submission_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Réentraîne le modèle choisi et crée le CSV de soumission Kaggle."
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
    run_prediction(config)


if __name__ == "__main__":
    main()
