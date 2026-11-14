# Référence des fichiers de configuration

Cette page décrit les clés JSON comprises par le projet, leurs valeurs acceptées et leur rôle. Le chargeur de configuration est strict, une clé inconnue à la racine, dans `preprocessing` ou dans `hyperparameter_search` provoque une erreur. Les fichiers d'exemple `config_*.json` montrent des expériences concrètes.

Les valeurs ci-dessous correspondent aux valeurs par défaut appliquées par le code lorsque la clé est omise.

## Clés générales

| Clé | Type et valeurs acceptées | Défaut | Description |
|---|---|---|---|
| `data_dir` | Chaîne non vide, chemin relatif ou absolu | `"data"` | Dossier contenant `train.csv`, `test.csv` et `sample_submission.csv`. |
| `artifacts_dir` | Chaîne non vide, chemin relatif ou absolu | `"artifacts"` | Dossier de sortie des modèles, métriques, prédictions de validation et résultats de recherche. |
| `submissions_dir` | Chaîne non vide, chemin relatif ou absolu | `"submissions"` | Dossier de sortie des fichiers de soumission. |
| `target` | Chaîne non vide, distincte de `id` | `"accepted"` | Nom de la cible dans `train.csv`. La classe positive attendue pour les probabilités est `1`. |
| `id` | Chaîne non vide, distincte de `target` | `"id"` | Nom de la colonne identifiant, conservée pour produire la soumission et exclue des variables explicatives. |
| `metric` | `"roc_auc"` | `"roc_auc"` | Métrique de validation et de recherche. Elle est calculée à partir des probabilités de la classe 1. |
| `model` | `logistic_regression`, `random_forest`, `extra_trees`, `xgboost`, `lightgbm`, `svm_soft_margin`, `svm_gaussian` | `"logistic_regression"` | Modèle à construire. Les deux variantes SVM utilisent `sklearn.svm.SVC` avec des noyaux imposés. |
| `model_params` | Objet JSON de paramètres du modèle | `{}` | Valeurs qui remplacent ou complètent les paramètres de projet listés plus bas. Les clés sont transmises au constructeur de l'estimateur choisi. |
| `validation_method` | `"holdout"` ou `"stratified_kfold"` | `"holdout"` | Méthode utilisée par `src.train` pour évaluer le modèle. La recherche d'hyperparamètres utilise toujours sa propre validation croisée stratifiée, réglée par `hyperparameter_search.cv`. |
| `validation_size` | Nombre entre 0 et 1, bornes exclues | `0.2` | Proportion réservée à la validation en mode `holdout`. Sans effet en validation croisée. |
| `n_splits` | Entier supérieur ou égal à 2 | `5` | Nombre de plis de `stratified_kfold`. Il est recommandé de ne pas dépasser l'effectif de la classe la moins représentée afin que chaque pli puisse contenir les deux classes. |
| `master_seed` | Entier de 0 à `2**32 - 1` | `42` | Graine de reproductibilité des découpages, recherches aléatoires et modèles. Ne pas ajouter `random_state` dans `model_params` : il est contrôlé par cette clé. |
| `preprocessing` | Objet JSON décrit dans la section [Prétraitement](#prétraitement) | `{}` | Options de transformation des variables; les valeurs internes par défaut sont indiquées ci-dessous. |
| `hyperparameter_search` | Objet JSON facultatif décrit dans la section [Recherche d'hyperparamètres](#recherche-dhyperparamètres) | absent | Lance une recherche lorsqu'on appelle `src.hyperparameter_search`. Sans ce bloc, la commande de recherche ne peut pas s'exécuter. |
| `run_name` | Chaîne facultative ne contenant que lettres ASCII, chiffres, `_` ou `-` | absent | Préfixe lisible donné au dossier d'exécution créé sous `artifacts/`. Un horodatage lui est ajouté. |
| `training_sample_size` | Entier supérieur ou égal à 2, facultatif; réservé aux deux SVM | absent | Taille d'un échantillon stratifié et reproductible de `train.csv`. Il doit être inférieur ou égal au nombre de lignes disponibles. Le même échantillonnage est utilisé par l'entraînement, la recherche et la génération de soumission. |


## Prétraitement

Le bloc accepte uniquement les clés suivantes. Les ajustements (imputations, normalisation et encodage) sont appris dans la pipeline sur les lignes d'entraînement du pli, afin d'éviter une fuite entre entraînement et validation.

| Clé | Type et valeurs acceptées | Défaut | Description |
|---|---|---|---|
| `method` | `"feature_engineering"` ou `"default"` | `"feature_engineering"` | `feature_engineering` retire `co-applicant_age`, regroupe les catégories de `debt_to_income_ratio`, plafonne le LTV au 99e percentile appris sur le pli, l'impute par la moyenne après plafonnement puis le convertit en trois classes (`≤80`, `>80–100`, `>100`). `property_value` et `income` sont imputés par la médiane et `loan_term` par 330. Le mode est appris pour `county_code`, `state_code`, `total_units`, `conforming_loan_limit`, les variables de race et d'ethnicité du demandeur et du co-demandeur, `applicant_age` et `debt_to_income_ratio`. Les valeurs catégorielles manquantes des autres colonnes sont représentées par `missing_category_token`. `default` ne fait pas ces transformations spécifiques et conserve les colonnes brutes. Les deux méthodes imputent les variables numériques restantes par la médiane et encodent les catégories en one-hot. |
| `scale_numeric` | Booléen | `true` | Ajoute une standardisation aux variables numériques après imputation. Recommandée pour les modèles sensibles à l'échelle, notamment la régression logistique et les SVM. |
| `add_missing_indicators` | Booléen | `true` | Ajoute aux variables numériques un indicateur lorsque la valeur était manquante. |
| `log1p_columns` | Liste de chaînes; liste vide autorisée | `[]` | Applique `log(1+x)` aux colonnes nommées. Les noms doivent appartenir au schéma numérique du `method` choisi et les valeurs observées doivent être positives ou nulles. Dans `default`, le schéma comprend `loan_amount`, `loan_to_value_ratio`, `property_value`, `income`, `loan_term`, `total_units`, `tract_population`, `tract_minority_population_percent`, `ffiec_msa_md_median_family_income`, `tract_to_msa_income_percentage`, `tract_owner_occupied_units`, `tract_one_to_four_family_homes` et `tract_median_age_of_housing_units`. Avec `feature_engineering`, `loan_to_value_ratio` est transformé en catégories et n'est pas disponible pour `log1p`. |
| `missing_category_token` | Chaîne non vide | `"__MISSING__"` | Texte représentant une valeur catégorielle manquante avant l'encodage one-hot. |

Dans les configurations actuelles, `log1p_columns` est vide : aucune variable n'est transformée par cette opération. Les variables sont encodées dans la pipeline et les catégories inconnues à la validation ou au test sont ignorées par le one-hot encoder.

## Modèles et paramètres usuels

Les paramètres ci-dessous sont les valeurs par défaut définies par le projet, avant application de `model_params`. Les valeurs possibles d'autres paramètres du constructeur dépendent de la version de scikit-learn, XGBoost ou LightGBM installée et peuvent aussi être passées dans `model_params`. Les paramètres numériques doivent respecter les contraintes de l'estimateur concerné; une combinaison incompatible peut être refusée à l'entraînement.

| `model` | Classe construite | Paramètres par défaut du projet dans `model_params` |
|---|---|---|
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` | `C: 1.0`, `max_iter: 2000`, `solver: "liblinear"` |
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` | `n_estimators: 300`, `max_depth: null`, `min_samples_leaf: 2`, `max_features: "sqrt"`, `n_jobs: 1` |
| `extra_trees` | `sklearn.ensemble.ExtraTreesClassifier` | `n_estimators: 300`, `max_depth: null`, `min_samples_leaf: 2`, `max_features: "sqrt"`, `n_jobs: 1` |
| `xgboost` | `xgboost.XGBClassifier` | `objective: "binary:logistic"`, `eval_metric: "logloss"`, `n_estimators: 500`, `learning_rate: 0.05`, `max_depth: 5`, `subsample: 0.8`, `colsample_bytree: 0.8`, `reg_lambda: 1.0`, `tree_method: "hist"`, `n_jobs: 1` |
| `lightgbm` | `lightgbm.LGBMClassifier` | `objective: "binary"`, `n_estimators: 500`, `learning_rate: 0.05`, `max_depth: 5`, `num_leaves: 31`, `subsample: 0.8`, `subsample_freq: 1`, `colsample_bytree: 0.8`, `reg_lambda: 1.0`, `n_jobs: 1` |
| `svm_soft_margin` | `sklearn.svm.SVC` | `C: 1.0`, `kernel: "linear"`, `probability: true` |
| `svm_gaussian` | `sklearn.svm.SVC` | `C: 1.0`, `gamma: "scale"`, `kernel: "rbf"`, `probability: true` |

Les principaux paramètres ajustables et leurs valeurs usuelles sont les suivants :

| Modèles | Paramètres | Valeurs usuelles et rôle |
|---|---|---|
| Régression logistique, SVM | `C` | Réel strictement positif; inverse de la force de régularisation. |
| Régression logistique | `max_iter` | Entier positif; nombre maximal d'itérations. `solver` sélectionne l'algorithme, par exemple `"liblinear"`, `"lbfgs"`, `"newton-cg"`, `"newton-cholesky"`, `"sag"` ou `"saga"` selon les compatibilités de la version installée et des pénalités choisies. |
| Forêts aléatoires, Extra Trees | `n_estimators` | Entier positif; nombre d'arbres. |
| Forêts aléatoires, Extra Trees | `max_depth` | Entier positif ou `null`; profondeur maximale, sans limite si `null`. |
| Forêts aléatoires, Extra Trees | `min_samples_leaf` | Entier positif ou fraction de l'effectif; nombre minimal d'observations dans une feuille. |
| Forêts aléatoires, Extra Trees | `max_features` | Entier, fraction, `"sqrt"`, `"log2"` ou `null`; nombre de variables candidates à chaque séparation (`null` utilise toutes les variables). |
| XGBoost, LightGBM | `n_estimators` | Entier positif; nombre d'arbres. |
| XGBoost, LightGBM | `learning_rate` | Réel positif; taux d'apprentissage. Les recherches du projet utilisent généralement une plage logarithmique. |
| XGBoost, LightGBM | `max_depth` | Profondeur maximale entière positive; LightGBM accepte aussi `-1` pour ne pas imposer de limite. |
| XGBoost, LightGBM | `subsample`, `colsample_bytree` | Réel entre 0 et 1; proportion d'observations ou de variables échantillonnées. |
| XGBoost, LightGBM | `reg_lambda` | Réel positif ou nul; régularisation L2. XGBoost expose aussi `min_child_weight` comme seuil de poids minimal d'une feuille. |
| LightGBM | `num_leaves`, `min_child_samples` | Entiers positifs; nombre maximal de feuilles et minimum d'observations par feuille. `subsample_freq` est un entier positif ou nul qui règle la fréquence du sous-échantillonnage. |


Deux contraintes sont imposées au SVM par le projet : `svm_soft_margin` conserve `kernel: "linear"`; `svm_gaussian` conserve `kernel: "rbf"`; les deux gardent `probability: true`, nécessaire aux probabilités ROC-AUC. `C` doit être positif; pour le SVM gaussien, `gamma` peut être `"scale"`, `"auto"` ou une valeur numérique positive.

## Recherche d'hyperparamètres

Le bloc `hyperparameter_search` accepte uniquement `method`, `parameters`, `cv`, `n_iter` et `n_jobs`.

| Clé | Type et valeurs acceptées | Défaut | Description |
|---|---|---|---|
| `method` | `"grid"` ou `"random"` | `"grid"` | `grid` évalue toutes les combinaisons des listes fournies. `random` tire des combinaisons selon les listes et distributions définies. |
| `parameters` | Objet JSON non vide, chaque valeur étant une liste non vide ou une distribution | Obligatoire | Noms de paramètres à tester. Les noms sans préfixe sont automatiquement appliqués au classifieur (`classifier__...`). Les noms préfixés par `classifier__` sont aussi acceptés. |
| `cv` | Entier supérieur ou égal à 2 | `3` | Nombre de plis d'une validation croisée stratifiée avec mélange. La graine `master_seed` fixe le découpage. |
| `n_iter` | Entier positif | `10` | Nombre de tirages pour `random`; sans effet sur `grid`. |
| `n_jobs` | Entier positif ou `-1`; `0` et valeurs inférieures à `-1` interdits | `1` | Nombre de processus parallèles utilisés par la recherche. `-1` demande l'utilisation de tous les cœurs disponibles. |

Une liste teste uniquement ses valeurs, par exemple `"max_depth": [10, 15, 20]`. Les distributions sont disponibles seulement avec `method: "random"` :

| Distribution | Format JSON | Valeurs tirées |
|---|---|---|
| `uniform` | `{"distribution": "uniform", "low": 0.6, "high": 1.0}` | Nombre réel uniforme entre `low` et `high`. Les bornes sont finies et `low < high`. |
| `loguniform` | `{"distribution": "loguniform", "low": 0.001, "high": 0.1}` | Nombre réel selon une loi uniforme en échelle logarithmique; les deux bornes doivent être positives et `low < high`. |
| `randint` | `{"distribution": "randint", "low": 3, "high": 20}` | Entier de `low` inclus à `high` exclu. Les deux bornes doivent être des entiers et `low < high`. |


### Exemple complet de recherche aléatoire

```json
{
  "model": "lightgbm",
  "model_params": {
    "n_estimators": 500,
    "n_jobs": 1
  },
  "hyperparameter_search": {
    "method": "random",
    "parameters": {
      "num_leaves": [31, 45, 60],
      "learning_rate": {
        "distribution": "loguniform",
        "low": 0.003,
        "high": 0.05
      },
      "n_estimators": {
        "distribution": "randint",
        "low": 300,
        "high": 1001
      },
      "colsample_bytree": {
        "distribution": "uniform",
        "low": 0.7,
        "high": 1.0
      }
    },
    "cv": 5,
    "n_iter": 40,
    "n_jobs": 1
  }
}
```

## Commandes

Depuis la racine du dépôt, sur Windows PowerShell :

```powershell
# Évaluer le modèle selon validation_method
.\.venv\Scripts\python.exe -m src.train --config config/config.json

# Lancer la recherche si hyperparameter_search est défini
.\.venv\Scripts\python.exe -m src.hyperparameter_search --config config/config_lightgbm_extended_random_search.json

# Créer une soumission avec le modèle configuré
.\.venv\Scripts\python.exe -m src.predict --config config/config.json
```

Un chemin de configuration non absolu est interprété depuis la racine du dépôt. La recherche et les autres commandes écrivent leurs résultats dans les dossiers configurés, en créant un répertoire horodaté par exécution.
