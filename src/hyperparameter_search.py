"""Recherche des hyperparamètres définis dans le fichier de configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.config import ExperimentConfig, load_config, new_run_name
from src.data import load_competition_data
from src.models import build_model_pipeline


def run_hyperparameter_search(config: ExperimentConfig) -> Path:
    """Évalue la grille configurée avec une pipeline ajustée dans chaque pli."""
    search_config = config.hyperparameter_search
    if search_config is None:
        raise ValueError("Ajoutez 'hyperparameter_search' au fichier de configuration.")

    data = load_competition_data(config)
    estimator = build_model_pipeline(config, data.features.columns)
    parameters = {
        name if "__" in name else f"classifier__{name}": values
        for name, values in search_config.parameters.items()
    }
    folds = StratifiedKFold(
        n_splits=search_config.cv,
        shuffle=True,
        random_state=config.master_seed,
    )
    common_options = {
        "estimator": estimator,
        "scoring": config.metric,
        "cv": folds,
        "n_jobs": search_config.n_jobs,
        "refit": False,
        "error_score": "raise",
    }
    if search_config.method == "grid":
        search = GridSearchCV(param_grid=parameters, **common_options)
    else:
        search = RandomizedSearchCV(
            param_distributions=parameters,
            n_iter=search_config.n_iter,
            random_state=config.master_seed,
            **common_options,
        )
    search.fit(data.features, data.target)

    run_directory = config.artifacts_dir / (
        "hyperparameter_search_" + new_run_name(config)
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config.config_path, run_directory / "config.json")
    pd.DataFrame(search.cv_results_).to_csv(
        run_directory / "search_results.csv", index=False
    )

    best_params = {
        name.removeprefix("classifier__"): value
        for name, value in search.best_params_.items()
    }
    metadata = {
        "model": config.model_name,
        "method": search_config.method,
        "scoring": config.metric,
        "cv": search_config.cv,
        "candidate_count": len(search.cv_results_["params"]),
        "best_score": float(search.best_score_),
        "best_params": best_params,
        "preprocessing_method": config.preprocessing.method,
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Meilleur ROC-AUC moyen : {search.best_score_:.6f}")
    print(f"Meilleurs paramètres : {best_params}")
    print(f"Résultats enregistrés dans : {run_directory}")
    return run_directory


def _parse_args() -> argparse.Namespace:
    """Définit les options de la commande de recherche."""
    parser = argparse.ArgumentParser(
        description="Recherche des hyperparamètres depuis le fichier de configuration."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Chemin vers le fichier de configuration JSON (défaut : config.json).",
    )
    return parser.parse_args()


def main() -> None:
    """Charge la configuration puis lance la recherche."""
    config = load_config(_parse_args().config)
    run_hyperparameter_search(config)


if __name__ == "__main__":
    main()
