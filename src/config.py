"""Lecture et validation de la configuration des expériences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_MODELS = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "svm_soft_margin",
    "svm_gaussian",
)
SVM_MODELS = ("svm_soft_margin", "svm_gaussian")

SEARCH_DISTRIBUTIONS = ("uniform", "loguniform", "randint")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Regroupe les options de prétraitement des variables."""

    scale_numeric: bool = True
    add_missing_indicators: bool = True
    log1p_columns: tuple[str, ...] = ()
    missing_category_token: str = "__MISSING__"
    method: str = "feature_engineering"


@dataclass(frozen=True)
class ParameterDistributionConfig:
    """Décrit une distribution numérique de recherche d'hyperparamètre."""

    distribution: str
    low: int | float
    high: int | float


@dataclass(frozen=True)
class HyperparameterSearchConfig:
    """Décrit les paramètres à explorer et le type de recherche."""

    method: str
    parameters: dict[str, list[Any] | ParameterDistributionConfig]
    cv: int
    n_iter: int
    n_jobs: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Regroupe les paramètres nécessaires à une expérience."""

    config_path: Path
    data_dir: Path
    artifacts_dir: Path
    submissions_dir: Path
    target_column: str
    id_column: str
    metric: str
    model_name: str
    model_params: dict[str, Any]
    validation_size: float
    master_seed: int
    preprocessing: PreprocessingConfig
    run_name: str | None = None
    validation_method: str = "holdout"
    n_splits: int = 5
    hyperparameter_search: HyperparameterSearchConfig | None = None
    training_sample_size: int | None = None


def _project_path(value: Any, field_name: str) -> Path:
    """Résout un chemin relatif depuis la racine du projet."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' doit être un chemin non vide.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_finite_number(value: Any) -> bool:
    """Indique si une valeur JSON est un nombre fini, sans accepter les booléens."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _parse_preprocessing(raw: Any) -> PreprocessingConfig:
    """Construit les options de prétraitement depuis le JSON."""
    if not isinstance(raw, dict):
        raise ValueError("'preprocessing' doit être un objet JSON.")

    allowed = {
        "scale_numeric",
        "add_missing_indicators",
        "log1p_columns",
        "missing_category_token",
        "method",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            "Options de prétraitement inconnues : " + ", ".join(sorted(unknown))
        )

    scale_numeric = raw.get("scale_numeric", True)
    add_missing = raw.get("add_missing_indicators", True)
    if not isinstance(scale_numeric, bool):
        raise ValueError("'preprocessing.scale_numeric' doit être booléen.")
    if not isinstance(add_missing, bool):
        raise ValueError("'preprocessing.add_missing_indicators' doit être booléen.")

    method = raw.get("method", "feature_engineering")
    if method not in ("feature_engineering", "default"):
        raise ValueError(
            "'preprocessing.method' doit être 'feature_engineering' ou 'default'."
        )

    log_columns = raw.get("log1p_columns", [])
    if not isinstance(log_columns, list) or any(
        not isinstance(column, str) for column in log_columns
    ):
        raise ValueError("'preprocessing.log1p_columns' doit être une liste de noms.")
    if len(log_columns) != len(set(log_columns)):
        raise ValueError("'preprocessing.log1p_columns' contient des doublons.")

    missing_token = raw.get("missing_category_token", "__MISSING__")
    if not isinstance(missing_token, str) or not missing_token:
        raise ValueError("'preprocessing.missing_category_token' doit être non vide.")

    return PreprocessingConfig(
        scale_numeric=scale_numeric,
        add_missing_indicators=add_missing,
        log1p_columns=tuple(log_columns),
        missing_category_token=missing_token,
        method=method,
    )


def _parse_hyperparameter_search(raw: Any) -> HyperparameterSearchConfig | None:
    """Lit les options facultatives de recherche d'hyperparamètres."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("'hyperparameter_search' doit être un objet JSON.")

    allowed = {"method", "parameters", "cv", "n_iter", "n_jobs"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            "Options de recherche inconnues : " + ", ".join(sorted(unknown))
        )

    method = raw.get("method", "grid")
    if method not in ("grid", "random"):
        raise ValueError("'hyperparameter_search.method' doit être 'grid' ou 'random'.")

    raw_parameters = raw.get("parameters")
    if not isinstance(raw_parameters, dict) or not raw_parameters:
        raise ValueError("'hyperparameter_search.parameters' doit être un objet non vide.")

    parameters: dict[str, list[Any] | ParameterDistributionConfig] = {}
    for name, values in raw_parameters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Chaque paramètre de recherche doit avoir un nom non vide.")
        if isinstance(values, list):
            if not values:
                raise ValueError(
                    "Chaque paramètre de recherche doit avoir une liste de valeurs non vide."
                )
            parameters[name] = values
            continue
        if not isinstance(values, dict):
            raise ValueError(
                "Chaque paramètre de recherche doit avoir une liste de valeurs ou une distribution."
            )
        if method == "grid":
            raise ValueError(
                "Les distributions ne sont prises en charge qu'avec method: 'random'."
            )

        allowed_distribution_fields = {"distribution", "low", "high"}
        unknown_fields = set(values) - allowed_distribution_fields
        if unknown_fields:
            raise ValueError(
                "Options de distribution inconnues : "
                + ", ".join(sorted(unknown_fields))
            )
        if set(values) != allowed_distribution_fields:
            raise ValueError(
                "Une distribution doit définir 'distribution', 'low' et 'high'."
            )

        family = values["distribution"]
        if not isinstance(family, str) or family not in SEARCH_DISTRIBUTIONS:
            raise ValueError(
                "Distribution inconnue : "
                f"{family!r}. Choix : {', '.join(SEARCH_DISTRIBUTIONS)}."
            )
        low = values["low"]
        high = values["high"]
        if not _is_finite_number(low) or not _is_finite_number(high):
            raise ValueError("Les bornes d'une distribution doivent être numériques et finies.")
        if family == "randint" and (
            not isinstance(low, int)
            or not isinstance(high, int)
            or isinstance(low, bool)
            or isinstance(high, bool)
        ):
            raise ValueError("Les bornes de la distribution 'randint' doivent être des entiers.")
        if low >= high:
            raise ValueError("La borne 'low' d'une distribution doit être inférieure à 'high'.")
        if family == "loguniform" and low <= 0:
            raise ValueError("La borne 'low' de 'loguniform' doit être strictement positive.")
        parameters[name] = ParameterDistributionConfig(
            distribution=family,
            low=low,
            high=high,
        )

    cv = raw.get("cv", 3)
    if isinstance(cv, bool) or not isinstance(cv, int) or cv < 2:
        raise ValueError("'hyperparameter_search.cv' doit être un entier supérieur ou égal à 2.")
    n_iter = raw.get("n_iter", 10)
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError("'hyperparameter_search.n_iter' doit être un entier positif.")
    n_jobs = raw.get("n_jobs", 1)
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, int)
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise ValueError("'hyperparameter_search.n_jobs' doit valoir -1 ou un entier positif.")

    return HyperparameterSearchConfig(
        method=method,
        parameters=dict(parameters),
        cv=cv,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Lit et valide un fichier de configuration JSON."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide dans {path} : {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("La configuration doit être un objet JSON.")

    allowed = {
        "data_dir",
        "artifacts_dir",
        "submissions_dir",
        "target",
        "id",
        "metric",
        "model",
        "model_params",
        "validation_method",
        "validation_size",
        "n_splits",
        "master_seed",
        "preprocessing",
        "hyperparameter_search",
        "run_name",
        "training_sample_size",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError("Clés de configuration inconnues : " + ", ".join(sorted(unknown)))

    target = raw.get("target", "accepted")
    identifier = raw.get("id", "id")
    if not isinstance(target, str) or not target:
        raise ValueError("'target' doit être une chaîne non vide.")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("'id' doit être une chaîne non vide.")
    if target == identifier:
        raise ValueError("'target' et 'id' doivent désigner deux colonnes distinctes.")

    metric = raw.get("metric", "roc_auc")
    if metric != "roc_auc":
        raise ValueError("La seule métrique prise en charge est 'roc_auc'.")

    model_name = raw.get("model", "logistic_regression")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Modèle inconnu : {model_name!r}. Choix : {', '.join(SUPPORTED_MODELS)}."
        )

    training_sample_size = raw.get("training_sample_size")
    if training_sample_size is not None:
        if (
            isinstance(training_sample_size, bool)
            or not isinstance(training_sample_size, int)
            or training_sample_size < 2
        ):
            raise ValueError("'training_sample_size' doit être un entier supérieur ou égal à 2.")
        if model_name not in SVM_MODELS:
            raise ValueError(
                "'training_sample_size' est réservé aux modèles SVM."
            )

    model_params = raw.get("model_params", {})
    if not isinstance(model_params, dict):
        raise ValueError("'model_params' doit être un objet JSON.")
    if "random_state" in model_params:
        raise ValueError(
            "Ne définissez pas 'random_state' dans model_params : "
            "master_seed le contrôle."
        )
    fixed_svm_params = {
        "svm_soft_margin": {"kernel": "linear", "probability": True},
        "svm_gaussian": {"kernel": "rbf", "probability": True},
    }.get(model_name, {})
    for name, expected in fixed_svm_params.items():
        configured_value = model_params.get(name, expected)
        matches = (
            type(configured_value) is bool and configured_value is expected
            if isinstance(expected, bool)
            else configured_value == expected
        )
        if name in model_params and not matches:
            raise ValueError(
                f"'{name}' est fixé à {expected!r} pour le modèle {model_name}."
            )

    validation_size = raw.get("validation_size", 0.2)
    if (
        isinstance(validation_size, bool)
        or not isinstance(validation_size, (int, float))
        or not 0.0 < float(validation_size) < 1.0
    ):
        raise ValueError("'validation_size' doit être strictement entre 0 et 1.")

    validation_method = raw.get("validation_method", "holdout")
    if validation_method not in ("holdout", "stratified_kfold"):
        raise ValueError(
            "'validation_method' doit être 'holdout' ou 'stratified_kfold'."
        )

    n_splits = raw.get("n_splits", 5)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("'n_splits' doit être un entier supérieur ou égal à 2.")

    master_seed = raw.get("master_seed", 42)
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or not 0 <= master_seed <= 2**32 - 1
    ):
        raise ValueError("'master_seed' doit être un entier entre 0 et 2**32 - 1.")

    run_name = raw.get("run_name")
    if run_name is not None and (
        not isinstance(run_name, str)
        or not run_name
        or not re.fullmatch(r"[A-Za-z0-9_-]+", run_name)
    ):
        raise ValueError("'run_name' ne peut contenir que lettres, chiffres, '_' et '-'.")

    hyperparameter_search = _parse_hyperparameter_search(
        raw.get("hyperparameter_search")
    )
    for parameter_name, values in (
        hyperparameter_search.parameters.items()
        if hyperparameter_search is not None
        else ()
    ):
        fixed_value = fixed_svm_params.get(
            parameter_name.removeprefix("classifier__")
        )
        if fixed_value is None:
            continue
        if not isinstance(values, list) or any(
            (
                type(value) is not bool or value is not fixed_value
                if isinstance(fixed_value, bool)
                else value != fixed_value
            )
            for value in values
        ):
            raise ValueError(
                f"Le paramètre de recherche '{parameter_name}' doit conserver "
                f"la valeur {fixed_value!r} pour le modèle {model_name}."
            )

    return ExperimentConfig(
        config_path=path,
        data_dir=_project_path(raw.get("data_dir", "data"), "data_dir"),
        artifacts_dir=_project_path(raw.get("artifacts_dir", "artifacts"), "artifacts_dir"),
        submissions_dir=_project_path(
            raw.get("submissions_dir", "submissions"), "submissions_dir"
        ),
        target_column=target,
        id_column=identifier,
        metric=metric,
        model_name=model_name,
        model_params=dict(model_params),
        validation_size=float(validation_size),
        master_seed=master_seed,
        preprocessing=_parse_preprocessing(raw.get("preprocessing", {})),
        validation_method=validation_method,
        n_splits=n_splits,
        hyperparameter_search=hyperparameter_search,
        run_name=run_name,
        training_sample_size=training_sample_size,
    )


def new_run_name(config: ExperimentConfig) -> str:
    """Crée un nom de dossier unique pour une exécution."""
    prefix = config.run_name or config.model_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}"
