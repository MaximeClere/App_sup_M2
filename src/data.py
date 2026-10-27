"""Lecture des données et découpage reproductible entraînement-validation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import ExperimentConfig
from src.preprocessing import NUMERIC_COLUMNS


NUMERIC_MISSING_VALUES = ["", "NA", "N/A", "Not Available", "Exempt", "Not Applicable"]


@dataclass
class CompetitionData:
    """Contient les données préparées pour l'entraînement et la prédiction."""

    features: pd.DataFrame
    target: pd.Series
    test_features: pd.DataFrame
    train_ids: pd.Series
    test_ids: pd.Series
    submission_template: pd.DataFrame


@dataclass
class TrainValidationSplit:
    """Contient les partitions d'entraînement et de validation."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    train_ids: pd.Series
    validation_ids: pd.Series


def _read_csv(path: Path, id_column: str) -> pd.DataFrame:
    """Lit un CSV en préservant les codes de catégorie « NA »."""
    # Préserve les codes de catégorie « NA » tout en reconnaissant les valeurs manquantes numériques.
    numeric_na_values = {
        column: NUMERIC_MISSING_VALUES for column in NUMERIC_COLUMNS
    }
    return pd.read_csv(
        path,
        dtype={id_column: "string"},
        keep_default_na=False,
        na_values=numeric_na_values,
    )


def load_competition_data(config: ExperimentConfig) -> CompetitionData:
    """Charge les fichiers CSV et sépare les identifiants, la cible et les variables."""
    id_column = config.id_column
    train = _read_csv(config.data_dir / "train.csv", id_column)
    test = _read_csv(config.data_dir / "test.csv", id_column)
    sample = _read_csv(config.data_dir / "sample_submission.csv", id_column)

    train_ids = train.pop(id_column).reset_index(drop=True)
    target = pd.to_numeric(train.pop(config.target_column))
    test_ids = test.pop(id_column).reset_index(drop=True)
    feature_columns = train.columns

    return CompetitionData(
        features=train,
        target=target.rename(config.target_column),
        test_features=test.loc[:, feature_columns].copy(),
        train_ids=train_ids,
        test_ids=test_ids,
        submission_template=sample.reset_index(drop=True),
    )


def split_train_validation(
    data: CompetitionData,
    validation_size: float,
    master_seed: int,
) -> TrainValidationSplit:
    """Crée un découpage stratifié et reproductible des données."""
    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        train_ids,
        validation_ids,
    ) = train_test_split(
        data.features,
        data.target,
        data.train_ids,
        test_size=validation_size,
        random_state=master_seed,
        shuffle=True,
        stratify=data.target,
    )

    return TrainValidationSplit(
        X_train=X_train,
        X_validation=X_validation,
        y_train=y_train,
        y_validation=y_validation,
        train_ids=train_ids.reset_index(drop=True),
        validation_ids=validation_ids.reset_index(drop=True),
    )


def iter_stratified_folds(
    data: CompetitionData,
    n_splits: int,
    master_seed: int,
) -> Iterator[TrainValidationSplit]:
    """Produit les partitions d'entraînement et de validation des plis."""
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=master_seed,
    )
    for train_indices, validation_indices in splitter.split(data.features, data.target):
        yield TrainValidationSplit(
            X_train=data.features.iloc[train_indices].reset_index(drop=True),
            X_validation=data.features.iloc[validation_indices].reset_index(drop=True),
            y_train=data.target.iloc[train_indices].reset_index(drop=True),
            y_validation=data.target.iloc[validation_indices].reset_index(drop=True),
            train_ids=data.train_ids.iloc[train_indices].reset_index(drop=True),
            validation_ids=data.train_ids.iloc[validation_indices].reset_index(drop=True),
        )


def build_submission(
    template: pd.DataFrame,
    test_ids: pd.Series,
    probabilities: np.ndarray,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Remplace les probabilités en respectant l'ordre du gabarit."""
    prediction_by_id = dict(zip(test_ids, probabilities))
    submission = template.copy()
    submission[target_column] = template[id_column].map(prediction_by_id)
    return submission
