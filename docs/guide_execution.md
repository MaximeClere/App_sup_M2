# Guide pratique : exploration, entraînement et soumission

Ce guide décrit le flux de travail de bout en bout sur Windows avec PowerShell. Toutes les commandes sont à lancer depuis la racine du dépôt, c’est-à-dire le dossier qui contient config.json et src/.

## 1. Préparer Python et les dépendances

Le projet est prévu pour Python 3.12. Depuis la racine du dépôt, installer uv une fois, puis synchroniser l'environnement avec les dépendances verrouillées :

~~~powershell
python -m pip install uv
python -m uv sync --locked
~~~

L’environnement virtuel est ignoré par Git. Pour l’activer dans une nouvelle fenêtre PowerShell :

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Si la politique PowerShell bloque l’activation, ne la modifiez pas : utilisez directement le chemin vers .venv\Scripts\python.exe dans les commandes ci-dessous.

## 2. Placer et protéger les données

Télécharger les fichiers de la compétition Kaggle et placer ces trois fichiers dans data/ :

- train.csv
- test.csv
- sample_submission.csv

Par défaut, data_dir vaut data dans config.json. Si les fichiers sont ailleurs, modifier cette valeur. Les chemins relatifs sont interprétés depuis la racine du dépôt.

Vérifier que Git ignore bien les CSV :

~~~powershell
git check-ignore -v data/train.csv data/test.csv data/sample_submission.csv
~~~

Ne jamais utiliser git add -f sur ces fichiers. Le .gitignore exclut tout le dossier data/.

## 3. Explorer les données

Lancer JupyterLab avec le notebook inclus :

~~~powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/exploration_donnees.ipynb
~~~

Dans JupyterLab, exécuter les cellules de haut en bas. Le notebook affiche les dimensions et types, la répartition de accepted, les valeurs manquantes, les distributions numériques, des variables catégorielles et des statistiques descriptives par modalité.

Le notebook utilise le chargeur commun du projet. Il ne conserve pas id comme variable explicative et n’utilise aucune cible du jeu test. Fermer JupyterLab avec Ctrl+C dans le terminal où il a été lancé.

## 4. Configurer une expérience

config.json contrôle une expérience. Il est possible de l’éditer directement ou d’en copier une pour garder des réglages distincts :

~~~powershell
Copy-Item config.json config-extra-trees.json
~~~

Dans le fichier de configuration choisi, régler notamment :

- model : logistic_regression, random_forest, extra_trees ou xgboost.
- model_params : paramètres scikit-learn compatibles avec le modèle choisi. Consulter les paramètres par défaut et le registre dans src/models.py. Quand le modèle change, remplacer aussi model_params par un objet adapté.
- master_seed : graine maîtresse utilisée pour le découpage et les modèles stochastiques.
- validation_method : holdout (défaut) ou stratified_kfold.
- validation_size : proportion réservée en mode holdout (0.2 par défaut).
- n_splits : nombre de plis en mode stratified_kfold (5 par défaut).
- preprocessing.scale_numeric : standardisation des variables continues.
- preprocessing.add_missing_indicators : indicateurs pour les variables numériques manquantes.
- preprocessing.log1p_columns : liste vide par défaut; ne renseigner que des colonnes continues numériques non négatives.

L’imputation médiane des variables continues, le traitement des catégories manquantes et l’encodage one-hot sont nécessaires au pipeline et restent actifs. id et accepted ne sont pas des variables explicatives. La métrique prise en charge est ROC-AUC.

Exemple de configuration compatible avec ExtraTrees :

~~~json
{
  "data_dir": "data",
  "artifacts_dir": "artifacts",
  "submissions_dir": "submissions",
  "target": "accepted",
  "id": "id",
  "metric": "roc_auc",
  "model": "extra_trees",
  "model_params": {
    "n_estimators": 300,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "n_jobs": -1
  },
  "validation_method": "holdout",
  "validation_size": 0.2,
  "n_splits": 5,
  "master_seed": 42,
  "preprocessing": {
    "scale_numeric": false,
    "add_missing_indicators": true,
    "log1p_columns": [],
    "missing_category_token": "__MISSING__"
  },
  "run_name": "extra_trees_base"
}
~~~

Enregistrer cette configuration sous config-extra-trees.json, puis la passer avec --config. Pour une comparaison propre, ne modifier qu’un petit nombre de choix à la fois et conserver les configurations testées.

Pour XGBoost, sélectionner `"model": "xgboost"`. Les paramètres par défaut sont définis dans src/models.py; ils peuvent être remplacés dans `model_params`. Exemple minimal :

~~~json
{
  "model": "xgboost",
  "model_params": {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 1
  }
}
~~~

## 5. Entraîner et évaluer sur validation

Avec la configuration par défaut :

~~~powershell
.\.venv\Scripts\python.exe -m src.train --config config.json
~~~

Avec une configuration alternative :

~~~powershell
.\.venv\Scripts\python.exe -m src.train --config config-extra-trees.json
~~~

Avec validation_method: "holdout", le programme crée un découpage stratifié à partir de master_seed et utilise validation_size pour choisir sa taille. Avec validation_method: "stratified_kfold", il entraîne et évalue un modèle sur chaque pli stratifié; n_splits indique le nombre de plis. Dans les deux modes, le prétraitement est ajusté uniquement sur les lignes d’entraînement de chaque partition, puis le ROC-AUC des probabilités de accepted = 1 est calculé. Chaque classe doit contenir au moins n_splits exemples pour la validation croisée.

Chaque exécution écrit un sous-dossier horodaté sous artifacts/. En mode holdout, il contient la configuration, validation_score, les prédictions et le modèle ajusté sur la partition d’entraînement. En mode stratified_kfold, metrics.json contient le ROC-AUC moyen dans validation_score et chaque score dans fold_validation_scores; aucun modèle ni fichier de prédictions par pli n’est enregistré. Le dossier artifacts/ est ignoré par Git.

## 6. Choisir un modèle et produire une soumission

Après avoir choisi les paramètres à partir des validations, lancer :

~~~powershell
.\.venv\Scripts\python.exe -m src.predict --config config.json
~~~

ou, pour la configuration sélectionnée :

~~~powershell
.\.venv\Scripts\python.exe -m src.predict --config config-extra-trees.json
~~~

Le pipeline est réentraîné sur toutes les lignes de train.csv, puis prédit les probabilités de la classe 1 sur test.csv. Le fichier créé dans submissions/ reprend exactement les colonnes et l’ordre du gabarit sample_submission.csv. Le modèle final et les métadonnées sont enregistrés dans artifacts/.

Ouvrir le CSV généré et vérifier les colonnes id,accepted, l’absence de valeurs manquantes, et que accepted contient des probabilités entre 0 et 1. Téléverser ensuite ce fichier dans la compétition Kaggle et conserver le score/nom de la soumission pour le rapport.

## 7. Conserver les résultats et le code

- Garder une copie distincte de chaque configuration de modèle testée.
- Noter pour chaque expérience le ROC-AUC de validation, le score Kaggle, les paramètres, les transformations et master_seed.
- Le répertoire data/, artifacts/ et les CSV de submissions/ sont exclus de Git par .gitignore.
- Le dépôt doit contenir le code, les configurations partageables, ce README et le notebook, mais pas les données ni les modèles entraînés.
- Pour la restitution, indiquer dans le rapport le lien du dépôt et la branche finale utilisée.

Le fichier src/hyperparameter_search.py est réservé à une prochaine étape; aucune recherche automatique par grille ou Optuna n’est implémentée à ce stade.

