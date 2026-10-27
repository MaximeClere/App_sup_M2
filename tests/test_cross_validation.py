"""Vérifications du découpage en plis et de ses artefacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.artifacts import save_cross_validation_artifacts, save_training_artifacts
from src.config import ExperimentConfig, PreprocessingConfig, load_config
from src.data import (
    CompetitionData,
    TrainValidationSplit,
    iter_stratified_folds,
)


class CrossValidationTests(unittest.TestCase):
    """Vérifie le découpage stratifié et l'enregistrement de ses scores."""

    def _data(self) -> CompetitionData:
        """Crée un petit jeu équilibré pour les vérifications."""
        features = pd.DataFrame({"value": range(20)})
        target = pd.Series([0] * 10 + [1] * 10, name="accepted")
        identifiers = pd.Series([f"id-{index}" for index in range(20)])
        return CompetitionData(
            features=features,
            target=target,
            test_features=features.iloc[:0],
            train_ids=identifiers,
            test_ids=identifiers.iloc[:0],
            submission_template=pd.DataFrame(columns=["id", "accepted"]),
        )

    def _config(
        self,
        root: Path,
        validation_method: str = "holdout",
        n_splits: int = 5,
    ) -> ExperimentConfig:
        """Crée une configuration temporaire pour les tests d'artefacts."""
        data_directory = root / "data"
        data_directory.mkdir(exist_ok=True)
        (data_directory / "train.csv").write_text("train", encoding="utf-8")
        config_path = root / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        return ExperimentConfig(
            config_path=config_path,
            data_dir=data_directory,
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
            validation_method=validation_method,
            n_splits=n_splits,
        )

    def test_folds_are_stratified_reproducible_and_cover_all_rows(self) -> None:
        """Vérifie que chaque ligne est validée une fois et que les plis sont stables."""
        data = self._data()
        folds = list(iter_stratified_folds(data, n_splits=5, master_seed=42))
        repeated_folds = list(iter_stratified_folds(data, n_splits=5, master_seed=42))

        validation_ids = [
            identifier for fold in folds for identifier in fold.validation_ids
        ]
        self.assertEqual(sorted(validation_ids), sorted(data.train_ids))
        self.assertEqual(
            [fold.validation_ids.tolist() for fold in folds],
            [fold.validation_ids.tolist() for fold in repeated_folds],
        )
        for fold in folds:
            self.assertFalse(set(fold.train_ids) & set(fold.validation_ids))
            self.assertEqual(fold.y_validation.value_counts().to_dict(), {0: 2, 1: 2})

    def test_config_defaults_to_holdout_and_rejects_too_few_splits(self) -> None:
        """Vérifie le mode par défaut et la borne minimale du nombre de plis."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            config = load_config(config_path)
            self.assertEqual(config.validation_method, "holdout")
            self.assertEqual(config.n_splits, 5)
            project_config = load_config(Path(__file__).parents[1] / "config.json")
            self.assertEqual(project_config.validation_method, "holdout")
            self.assertEqual(project_config.n_splits, 5)

            config_path.write_text(
                json.dumps(
                    {"validation_method": "stratified_kfold", "n_splits": 3}
                ),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.validation_method, "stratified_kfold")
            self.assertEqual(config.n_splits, 3)

            config_path.write_text(json.dumps({"n_splits": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "n_splits"):
                load_config(config_path)

    def test_cross_validation_artifact_stores_mean_and_fold_scores(self) -> None:
        """Vérifie que les artefacts contiennent la moyenne et les scores individuels."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._config(root, "stratified_kfold", 3)
            data = self._data()
            model = SimpleNamespace(
                named_steps={
                    "classifier": SimpleNamespace(
                        get_params=lambda deep=False: {"random_state": 42}
                    )
                }
            )
            scores = [0.7, 0.8, 0.9]

            run_directory = save_cross_validation_artifacts(
                config=config,
                data=data,
                model=model,
                fold_scores=scores,
            )
            metadata = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))

            self.assertAlmostEqual(metadata["validation_score"], 0.8)
            self.assertEqual(metadata["fold_validation_scores"], scores)
            self.assertFalse((run_directory / "validation_model.joblib").exists())
            self.assertFalse((run_directory / "validation_predictions.csv").exists())

    def test_holdout_artifact_keeps_model_and_predictions(self) -> None:
        """Vérifie que le mode holdout conserve ses artefacts actuels."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = self._config(root)
            data = self._data()
            validation_indices = [0, 1, 10, 11]
            train_indices = [index for index in range(20) if index not in validation_indices]
            split = TrainValidationSplit(
                X_train=data.features.iloc[train_indices],
                X_validation=data.features.iloc[validation_indices],
                y_train=data.target.iloc[train_indices],
                y_validation=data.target.iloc[validation_indices],
                train_ids=data.train_ids.iloc[train_indices],
                validation_ids=data.train_ids.iloc[validation_indices],
            )
            model = Pipeline([("classifier", LogisticRegression())])
            run_directory = save_training_artifacts(
                config=config,
                data=data,
                split=split,
                model=model,
                probabilities=np.array([0.1, 0.2, 0.8, 0.9]),
                score=0.9,
            )

            self.assertTrue((run_directory / "validation_model.joblib").exists())
            self.assertTrue((run_directory / "validation_predictions.csv").exists())
            metadata = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["validation_method"], "holdout")
            self.assertEqual(metadata["validation_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
