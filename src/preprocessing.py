"""Leakage-safe preprocessing pipelines for mixed tabular data."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PreprocessingConfig


# Continuous measurements from the challenge schema. All remaining predictors
# are treated as categories, including numeric-looking codes and geography IDs.
NUMERIC_COLUMNS = (
    "loan_amount",
    "loan_to_value_ratio",
    "property_value",
    "income",
    "loan_term",
    "total_units",
    "tract_population",
    "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
)


def _category_as_string(value: object, missing_token: str) -> str:
    """Normalize missing and numeric category values across train and test."""
    if pd.isna(value):
        return missing_token
    if isinstance(value, str) and not value.strip():
        return missing_token
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    return str(value).strip()


class CategoryStringTransformer(BaseEstimator, TransformerMixin):
    """Convert every category to a stable string and encode missingness explicitly."""

    def __init__(self, missing_token: str = "__MISSING__") -> None:
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: object = None) -> CategoryStringTransformer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Les variables catégorielles doivent être un DataFrame pandas.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Les colonnes catégorielles diffèrent de celles ajustées.")

        transformed = X.copy()
        for column in transformed.columns:
            transformed[column] = [
                _category_as_string(value, self.missing_token)
                for value in transformed[column]
            ]
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log(1+x) to selected non-negative numeric columns."""

    def __init__(self, columns: tuple[str, ...] = ()) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> Log1pTransformer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Le prétraitement log1p attend un DataFrame pandas.")
        missing = set(self.columns) - set(X.columns)
        if missing:
            raise ValueError(
                "Colonnes log1p absentes des variables : " + ", ".join(sorted(missing))
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Les colonnes diffèrent de celles utilisées pour ajuster log1p.")

        transformed = X.copy()
        for column in self.columns:
            try:
                values = pd.to_numeric(transformed[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"La colonne log1p '{column}' doit contenir des nombres."
                ) from exc
            if (values.dropna() < 0).any():
                raise ValueError(
                    f"log1p n'est défini ici que pour des valeurs positives; "
                    f"'{column}' contient une valeur négative."
                )
            transformed[column] = np.log1p(values)
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


def build_preprocessor(
    feature_columns: Sequence[str],
    config: PreprocessingConfig,
) -> Pipeline:
    """Build transformations that are fitted only on the model's training rows."""
    feature_columns = list(feature_columns)
    missing_numeric = set(NUMERIC_COLUMNS) - set(feature_columns)
    if missing_numeric:
        raise ValueError(
            "Schéma du challenge incomplet; variables numériques absentes : "
            + ", ".join(sorted(missing_numeric))
        )

    invalid_log_columns = set(config.log1p_columns) - set(NUMERIC_COLUMNS)
    if invalid_log_columns:
        raise ValueError(
            "log1p_columns doit uniquement contenir des variables numériques connues : "
            + ", ".join(sorted(invalid_log_columns))
        )

    numeric_columns = [name for name in feature_columns if name in NUMERIC_COLUMNS]
    categorical_columns = [name for name in feature_columns if name not in NUMERIC_COLUMNS]

    numeric_steps = [
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=config.add_missing_indicators,
            keep_empty_features=True,
        ))
    ]
    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("normalize", CategoryStringTransformer(config.missing_category_token)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    transformers = [
        ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]
    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("log1p", Log1pTransformer(config.log1p_columns)),
            ("columns", column_transformer),
        ]
    )
