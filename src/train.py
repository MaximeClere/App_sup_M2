"""Entraîne un modèle configuré et l'évalue sur un jeu de validation stratifié."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from statistics import mean

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.artifacts import save_cross_validation_artifacts, save_training_artifacts
from src.config import ExperimentConfig, load_config
from src.data import (
    TrainValidationSplit,
    iter_stratified_folds,
    load_competition_data,
    split_train_validation,
)
from src.models import build_model_pipeline, positive_class_probabilities


def _seed_global_generators(seed: int) -> None:
    """Fixe les graines des générateurs aléatoires utilisés."""
    random.seed(seed)
    np.random.seed(seed)


def _fit_and_score(
    config: ExperimentConfig,
    split: TrainValidationSplit,
) -> tuple[Pipeline, float, np.ndarray]:
    """Entraîne une pipeline sur un pli et calcule son ROC-AUC."""
    model = build_model_pipeline(config, split.X_train.columns)
    model.fit(split.X_train, split.y_train)
    probabilities = positive_class_probabilities(model, split.X_validation)
    score = float(roc_auc_score(split.y_validation, probabilities))
    return model, score, probabilities


def run_training(config: ExperimentConfig) -> Path:
    """Évalue le modèle selon la méthode de validation configurée."""
    _seed_global_generators(config.master_seed)
    data = load_competition_data(config)

    if config.validation_method == "holdout":
        split = split_train_validation(
            data=data,
            validation_size=config.validation_size,
            master_seed=config.master_seed,
        )
        model, score, probabilities = _fit_and_score(config, split)
        run_directory = save_training_artifacts(
            config=config,
            data=data,
            split=split,
            model=model,
            probabilities=probabilities,
            score=score,
        )
        print(f"ROC-AUC validation : {score:.6f}")
    else:
        fold_scores = []
        for fold_number, split in enumerate(
            iter_stratified_folds(data, config.n_splits, config.master_seed),
            start=1,
        ):
            model, score, _ = _fit_and_score(config, split)
            fold_scores.append(score)
            print(f"ROC-AUC pli {fold_number} : {score:.6f}")

        run_directory = save_cross_validation_artifacts(
            config=config,
            data=data,
            model=model,
            fold_scores=fold_scores,
        )
        print(f"ROC-AUC validation moyen : {mean(fold_scores):.6f}")

    print(f"Modèle : {config.model_name}")
    print(f"Résultats enregistrés dans : {run_directory}")
    return run_directory


def _parse_args() -> argparse.Namespace:
    """Définit et lit les arguments de la ligne de commande."""
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
    """Charge la configuration puis lance l'entraînement."""
    args = _parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
