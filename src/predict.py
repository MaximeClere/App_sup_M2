"""Réentraîne le modèle sélectionné et crée une soumission Kaggle."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from src.artifacts import save_prediction_artifacts
from src.config import ExperimentConfig, load_config
from src.data import build_submission, load_competition_data
from src.models import build_model_pipeline, positive_class_probabilities


def _seed_global_generators(seed: int) -> None:
    """Fixe les graines des générateurs aléatoires utilisés."""
    random.seed(seed)
    np.random.seed(seed)


def run_prediction(config: ExperimentConfig) -> Path:
    """Réentraîne le modèle sur toutes les données étiquetées et prédit le test."""
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

    submission_path, model_path = save_prediction_artifacts(
        config=config,
        data=data,
        model=model,
        submission=submission,
    )

    print(f"Modèle réentraîné sur {len(data.features)} lignes.")
    print(f"Soumission enregistrée dans : {submission_path}")
    print(f"Modèle enregistré dans : {model_path}")
    return submission_path


def _parse_args() -> argparse.Namespace:
    """Définit et lit les arguments de la ligne de commande."""
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
    """Charge la configuration puis lance la prédiction."""
    args = _parse_args()
    config = load_config(args.config)
    run_prediction(config)


if __name__ == "__main__":
    main()
