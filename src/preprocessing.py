"""Prétraitement sans fuite de données pour des variables tabulaires mixtes."""

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


# Mesures continues du schéma brut; les autres variables explicatives sont catégorielles.
LTV_COLUMN = "loan_to_value_ratio"
NUMERIC_COLUMNS = (
    "loan_amount",
    LTV_COLUMN,
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

ENGINEERED_NUMERIC_COLUMNS = tuple(
    column for column in NUMERIC_COLUMNS if column != LTV_COLUMN
)
DROP_COLUMNS = ("co-applicant_age",)
MODE_IMPUTE_COLUMNS = (
    "county_code",
    "state_code",
    "total_units",
    "conforming_loan_limit",
    "applicant_ethnicity-1",
    "applicant_race-1",
    "co-applicant_ethnicity-1",
    "co-applicant_race-1",
    "applicant_age",
    "debt_to_income_ratio",
)


def _category_as_string(value: object, missing_token: str) -> str:
    """Normalise les valeurs catégorielles manquantes ou numériques."""
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


def _group_debt_to_income(value: object) -> object:
    """Regroupe les catégories DTI comprises entre 20 % et 49 %."""
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return pd.NA

    category = _category_as_string(value, "__MISSING__")
    if category in {"20%-<30%", "30%-<36%"}:
        return "20%-49%"
    try:
        numeric_value = float(category)
    except ValueError:
        return category
    if numeric_value.is_integer() and 20 <= numeric_value <= 49:
        return "20%-49%"
    return category


def _category_for_mode(value: object) -> object:
    """Normalise une catégorie sans confondre le vide avec un vrai code."""
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return pd.NA
    return _category_as_string(value, "__MISSING__")


def _prepare_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Retire la variable exclue et regroupe les catégories déterministes."""
    prepared = X.drop(columns=list(DROP_COLUMNS), errors="ignore").copy()
    if "debt_to_income_ratio" in prepared:
        prepared["debt_to_income_ratio"] = prepared["debt_to_income_ratio"].map(
            _group_debt_to_income
        )
    return prepared


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Apprend les imputations et transforme les variables du projet."""

    def __init__(self) -> None:
        """Initialise le transformeur sans statistique ajustée."""

    def fit(
        self, X: pd.DataFrame, y: object = None
    ) -> FeatureEngineeringTransformer:
        """Apprend les statistiques uniquement sur les lignes reçues."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Les variables doivent être un DataFrame pandas.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        prepared = _prepare_engineered_features(X)

        ltv = pd.to_numeric(prepared[LTV_COLUMN].replace("", pd.NA), errors="coerce")
        observed_ltv = ltv.dropna()
        self.ltv_cap_ = float(observed_ltv.quantile(0.99))
        self.ltv_mean_ = float(observed_ltv.clip(upper=self.ltv_cap_).mean())

        self.numeric_medians_ = {
            column: float(
                pd.to_numeric(prepared[column].replace("", pd.NA), errors="coerce").median()
            )
            for column in ("property_value", "income")
        }

        self.mode_imputations_: dict[str, object] = {}
        for column in MODE_IMPUTE_COLUMNS:
            if column not in prepared:
                continue
            if column == "total_units":
                values = pd.to_numeric(
                    prepared[column].replace("", pd.NA), errors="coerce"
                )
            else:
                values = prepared[column].map(_category_for_mode)
            modes = values.mode(dropna=True)
            self.mode_imputations_[column] = modes.iloc[0] if not modes.empty else "__MISSING__"

        self.statistics_ = {
            "loan_to_value_ratio": {
                "cap_p99": self.ltv_cap_,
                "mean_after_cap": self.ltv_mean_,
            },
            "property_value_median": self.numeric_medians_["property_value"],
            "income_median": self.numeric_medians_["income"],
            "loan_term_fill": 330,
            "mode_imputations": self.mode_imputations_.copy(),
        }
        self.feature_names_out_ = np.asarray(prepared.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applique les statistiques apprises en conservant un schéma commun."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Les colonnes diffèrent de celles utilisées pour l'ajustement.")

        transformed = _prepare_engineered_features(X)
        ltv = pd.to_numeric(
            transformed[LTV_COLUMN].replace("", pd.NA), errors="coerce"
        ).fillna(self.ltv_mean_)
        ltv_bins = pd.cut(
            ltv,
            bins=[-np.inf, 80, 100, np.inf],
            labels=["≤80", ">80–100", ">100"],
        )
        transformed[LTV_COLUMN] = ltv_bins.astype(object)

        for column, median in self.numeric_medians_.items():
            values = pd.to_numeric(
                transformed[column].replace("", pd.NA), errors="coerce"
            )
            transformed[column] = values.fillna(median)

        transformed["loan_term"] = pd.to_numeric(
            transformed["loan_term"].replace("", pd.NA), errors="coerce"
        ).fillna(330)

        for column, mode in self.mode_imputations_.items():
            if column == "total_units":
                values = pd.to_numeric(
                    transformed[column].replace("", pd.NA), errors="coerce"
                )
            else:
                values = transformed[column].map(_category_for_mode)
            transformed[column] = values.fillna(mode)
        return transformed

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Retourne les noms des colonnes après suppression de la variable."""
        return self.feature_names_out_


class CategoryStringTransformer(BaseEstimator, TransformerMixin):
    """Convertit les catégories en chaînes et marque les valeurs manquantes."""

    def __init__(self, missing_token: str = "__MISSING__") -> None:
        """Définit le texte qui représentera les valeurs manquantes."""
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: object = None) -> CategoryStringTransformer:
        """Mémorise les colonnes catégorielles à transformer."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Les variables catégorielles doivent être un DataFrame pandas.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convertit les valeurs catégorielles en chaînes normalisées."""
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
        """Retourne les noms des colonnes catégorielles conservées."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Applique log(1+x) aux colonnes numériques sélectionnées."""

    def __init__(self, columns: tuple[str, ...] = ()) -> None:
        """Mémorise les colonnes auxquelles appliquer log(1+x)."""
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> Log1pTransformer:
        """Vérifie et mémorise les colonnes d'entrée."""
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
        """Transforme les colonnes sélectionnées après vérification des valeurs."""
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
        """Retourne les noms des colonnes conservées après transformation."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


def build_preprocessor(
    feature_columns: Sequence[str],
    config: PreprocessingConfig,
) -> Pipeline:
    """Construit le prétraitement ajusté uniquement sur les lignes d'entraînement."""
    feature_columns = list(feature_columns)
    missing_numeric = set(NUMERIC_COLUMNS) - set(feature_columns)
    if missing_numeric:
        raise ValueError(
            "Schéma du challenge incomplet; variables numériques absentes : "
            + ", ".join(sorted(missing_numeric))
        )

    numeric_schema = (
        NUMERIC_COLUMNS
        if config.method == "default"
        else ENGINEERED_NUMERIC_COLUMNS
    )
    invalid_log_columns = set(config.log1p_columns) - set(numeric_schema)
    if invalid_log_columns:
        raise ValueError(
            "log1p_columns doit uniquement contenir des variables numériques connues : "
            + ", ".join(sorted(invalid_log_columns))
        )

    if config.method == "feature_engineering":
        feature_columns = [name for name in feature_columns if name not in DROP_COLUMNS]
    numeric_columns = [name for name in feature_columns if name in numeric_schema]
    categorical_columns = [name for name in feature_columns if name not in numeric_schema]

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

    steps = []
    if config.method == "feature_engineering":
        steps.append(("feature_engineering", FeatureEngineeringTransformer()))
    steps.extend(
        [
            ("log1p", Log1pTransformer(config.log1p_columns)),
            ("columns", column_transformer),
        ]
    )
    return Pipeline(steps=steps)
