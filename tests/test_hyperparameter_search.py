"""Vérifications de la configuration et de la recherche d'hyperparamètres."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    ExperimentConfig,
    HyperparameterSearchConfig,
    PreprocessingConfig,
    load_config,
)
from src.data import CompetitionData
from src.hyperparameter_search import run_hyperparameter_search


class HyperparameterSearchTests(unittest.TestCase):
    """Vérifie les méthodes de recherche et leurs résultats sauvegardés."""

    def _data(self) -> CompetitionData:
        """Crée un petit jeu binaire pour tester la recherche sans données externes."""
        features = pd.DataFrame({"signal": np.r_[-20:-1, 1:20], "noise": np.arange(38)})
        target = pd.Series([0] * 19 + [1] * 19, name="accepted")
        identifiers = pd.Series([f"id-{i}" for i in range(len(target))])
        return CompetitionData(
            features=features,
            target=target,
            test_features=features.iloc[:0],
            train_ids=identifiers,
            test_ids=identifiers.iloc[:0],
            submission_template=pd.DataFrame(columns=["id", "accepted"]),
        )

    def _config(
        self, root: Path, method: str
    ) -> ExperimentConfig:
        """Prépare une configuration temporaire de recherche."""
        config_path = root / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        return ExperimentConfig(
            config_path=config_path,
            data_dir=root / "data",
            artifacts_dir=root / "artifacts",
            submissions_dir=root / "submissions",
            target_column="accepted",
            id_column="id",
            metric="roc_auc",
            model_name="logistic_regression",
            model_params={},
            validation_size=0.2,
            master_seed=42,
            preprocessing=PreprocessingConfig(),
            hyperparameter_search=HyperparameterSearchConfig(
                method=method,
                parameters={"C": [0.1, 1.0, 10.0]},
                cv=2,
                n_iter=2,
                n_jobs=1,
            ),
        )

    def _run(self, root: Path, method: str) -> Path:
        """Lance la recherche avec un classifieur léger de test."""
        config = self._config(root, method)
        pipeline = Pipeline(
            [("classifier", LogisticRegression(solver="liblinear", random_state=42))]
        )
        with (
            patch("src.hyperparameter_search.load_competition_data", return_value=self._data()),
            patch("src.hyperparameter_search.build_model_pipeline", return_value=pipeline),
        ):
            return run_hyperparameter_search(config)

    def test_grid_search_saves_scores_and_best_parameters(self) -> None:
        """Vérifie la grille, la moyenne CV et les fichiers produits."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = self._run(Path(temporary_directory), "grid")
            metadata = json.loads(
                (run_directory / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["candidate_count"], 3)
            self.assertIn("C", metadata["best_params"])
            self.assertTrue((run_directory / "search_results.csv").is_file())
            self.assertTrue((run_directory / "config.json").is_file())

    def test_random_search_respects_iteration_count(self) -> None:
        """Vérifie que la recherche aléatoire tire le nombre configuré de candidats."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = self._run(Path(temporary_directory), "random")
            metadata = json.loads(
                (run_directory / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["method"], "random")
            self.assertEqual(metadata["candidate_count"], 2)

    def test_config_loads_search_settings_and_rejects_empty_candidates(self) -> None:
        """Vérifie le chargement et la validation de la grille dans le JSON."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text(
                json.dumps({
                    "hyperparameter_search": {
                        "method": "grid",
                        "parameters": {"max_depth": [3, 5]},
                        "cv": 4,
                    }
                }),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.hyperparameter_search.method, "grid")
            self.assertEqual(config.hyperparameter_search.cv, 4)

            path.write_text(
                json.dumps({
                    "hyperparameter_search": {
                        "parameters": {"max_depth": []}
                    }
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "liste de valeurs"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
