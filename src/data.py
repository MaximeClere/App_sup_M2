"""Data loading, schema validation, and reproducible train/validation splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ExperimentConfig
from src.preprocessing import NUMERIC_COLUMNS


NUMERIC_MISSING_VALUES = ["", "NA", "N/A", "Not Available", "Exempt", "Not Applicable"]


@dataclass
class CompetitionData:
    features: pd.DataFrame
    target: pd.Series
    test_features: pd.DataFrame
    train_ids: pd.Series
    test_ids: pd.Series
    submission_template: pd.DataFrame


@dataclass
class TrainValidationSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    train_ids: pd.Series
    validation_ids: pd.Series


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier requis introuvable : {path}. "
            "Vérifiez data_dir dans le fichier de configuration."
        )


def _read_csv(path: Path, id_column: str) -> pd.DataFrame:
    # Preserve literal "NA" category codes (for example, not-applicable values)
    # while still recognizing explicit numeric missing-value markers.
    numeric_na_values = {
        column: NUMERIC_MISSING_VALUES for column in NUMERIC_COLUMNS
    }
    return pd.read_csv(
        path,
        dtype={id_column: "string"},
        keep_default_na=False,
        na_values=numeric_na_values,
    )


def _coerce_numeric_columns(frame: pd.DataFrame, source_name: str) -> None:
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            raise ValueError(
                f"{source_name} ne contient pas la variable numérique '{column}'."
            )
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_name}.{column} contient une valeur qui n'est pas numérique "
                "et ne fait pas partie des marqueurs de données manquantes reconnus."
            ) from exc


def _canonical_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def _validate_unique_ids(values: pd.Series, name: str) -> list[str]:
    if values.isna().any():
        raise ValueError(f"La colonne d'identifiant '{name}' contient des valeurs manquantes.")
    keys = [_canonical_id(value) for value in values]
    if any(not key for key in keys):
        raise ValueError(f"La colonne d'identifiant '{name}' contient un identifiant vide.")
    if len(keys) != len(set(keys)):
        raise ValueError(f"La colonne d'identifiant '{name}' contient des doublons.")
    return keys


def load_competition_data(config: ExperimentConfig) -> CompetitionData:
    """Load the local competition CSV files and enforce their expected schema."""
    paths = {
        "train": config.data_dir / "train.csv",
        "test": config.data_dir / "test.csv",
        "sample_submission": config.data_dir / "sample_submission.csv",
    }
    for path in paths.values():
        _require_file(path)

    train = _read_csv(paths["train"], config.id_column)
    test = _read_csv(paths["test"], config.id_column)
    sample = _read_csv(paths["sample_submission"], config.id_column)

    id_column = config.id_column
    target_column = config.target_column
    if id_column not in train.columns or target_column not in train.columns:
        raise ValueError(
            f"train.csv doit contenir les colonnes '{id_column}' et '{target_column}'."
        )
    if id_column not in test.columns:
        raise ValueError(f"test.csv ne contient pas la colonne d'identifiant '{id_column}'.")
    if target_column in test.columns:
        raise ValueError(
            f"test.csv contient la cible '{target_column}', alors qu'elle doit être absente."
        )
    if id_column not in sample.columns or target_column not in sample.columns:
        raise ValueError(
            f"sample_submission.csv doit contenir '{id_column}' et '{target_column}'."
        )
    if list(sample.columns) != [id_column, target_column]:
        raise ValueError(
            "Le gabarit de soumission doit contenir exactement les colonnes "
            f"dans cet ordre : {id_column}, {target_column}."
        )
    if train.empty or test.empty:
        raise ValueError("train.csv et test.csv doivent contenir au moins une ligne.")

    train_ids = train[id_column].copy()
    test_ids = test[id_column].copy()
    train_id_keys = _validate_unique_ids(train_ids, f"train.{id_column}")
    test_id_keys = _validate_unique_ids(test_ids, f"test.{id_column}")
    sample_id_keys = _validate_unique_ids(sample[id_column], f"sample_submission.{id_column}")

    if len(sample_id_keys) != len(test_id_keys) or set(sample_id_keys) != set(test_id_keys):
        raise ValueError(
            "Les identifiants du gabarit de soumission ne correspondent pas à ceux de test.csv."
        )
    if set(train_id_keys).intersection(test_id_keys):
        raise ValueError("Les identifiants de train.csv et test.csv doivent être disjoints.")

    raw_target = pd.to_numeric(train[target_column], errors="coerce")
    if raw_target.isna().any():
        raise ValueError(f"La cible '{target_column}' contient des valeurs manquantes ou invalides.")
    observed_classes = set(raw_target.unique().tolist())
    if not observed_classes.issubset({0, 1}) or observed_classes != {0, 1}:
        raise ValueError(
            f"La cible '{target_column}' doit contenir exactement les classes 0 et 1; "
            f"valeurs trouvées : {sorted(observed_classes)}."
        )

    feature_columns = [
        column for column in train.columns if column not in {id_column, target_column}
    ]
    test_feature_columns = [column for column in test.columns if column != id_column]
    missing_from_test = set(feature_columns) - set(test_feature_columns)
    extra_in_test = set(test_feature_columns) - set(feature_columns)
    if missing_from_test or extra_in_test:
        raise ValueError(
            "Les variables explicatives de train.csv et test.csv ne correspondent pas. "
            f"Absentes du test : {sorted(missing_from_test)}; "
            f"supplémentaires dans le test : {sorted(extra_in_test)}."
        )

    _coerce_numeric_columns(train, "train.csv")
    _coerce_numeric_columns(test, "test.csv")

    return CompetitionData(
        features=train.loc[:, feature_columns].copy(),
        target=raw_target.astype("int8").rename(target_column),
        test_features=test.loc[:, feature_columns].copy(),
        train_ids=train_ids.reset_index(drop=True),
        test_ids=test_ids.reset_index(drop=True),
        submission_template=sample.reset_index(drop=True),
    )


def split_train_validation(
    data: CompetitionData,
    validation_size: float,
    master_seed: int,
) -> TrainValidationSplit:
    """Create a stratified, deterministic holdout split."""
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


def build_submission(
    template: pd.DataFrame,
    test_ids: pd.Series,
    probabilities: np.ndarray,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Place class-1 probabilities in template order and validate the result."""
    if len(test_ids) != len(probabilities) or len(template) != len(test_ids):
        raise ValueError("Le nombre d'identifiants, de probabilités et de lignes doit coïncider.")
    if list(template.columns) != [id_column, target_column]:
        raise ValueError("Le gabarit de soumission ne respecte pas le schéma attendu.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Les prédictions contiennent des valeurs non finies.")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Les probabilités doivent être comprises entre 0 et 1.")

    prediction_by_id = {
        _canonical_id(identifier): float(probability)
        for identifier, probability in zip(test_ids, probabilities, strict=True)
    }
    template_keys = [_canonical_id(identifier) for identifier in template[id_column]]
    if len(prediction_by_id) != len(test_ids) or set(prediction_by_id) != set(template_keys):
        raise ValueError("Les identifiants de prédiction ne correspondent pas au gabarit.")

    submission = template.copy()
    submission[target_column] = [prediction_by_id[key] for key in template_keys]
    return submission
