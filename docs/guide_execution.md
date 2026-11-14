# Guide pratique : exploration, entraînement et soumission

Ce guide décrit le flux de travail de bout en bout sur Windows avec PowerShell. Toutes les commandes sont à lancer depuis la racine du dépôt, c’est-à-dire le dossier qui contient src/.

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

Si la politique PowerShell bloque l’activation, ne la modifiez pas. Utilisez directement le chemin vers .venv\Scripts\python.exe dans les commandes ci-dessous.

## 2. Placer les données

Télécharger les fichiers de la compétition Kaggle et placer ces trois fichiers dans data/ :

- train.csv
- test.csv
- sample_submission.csv

Par défaut, data_dir vaut data dans config.json. Si les fichiers sont ailleurs, modifier cette valeur. Les chemins relatifs sont interprétés depuis la racine du dépôt.

## 3. Explorer les données

Lancer JupyterLab avec le notebook inclus :

~~~powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/exploration_donnees.ipynb
~~~

Dans JupyterLab, exécuter les cellules de haut en bas. Le notebook affiche les dimensions et types, la répartition de accepted, les valeurs manquantes, les distributions numériques, des variables catégorielles et des statistiques descriptives par modalité.

## 4. Configurer une expérience

config.json contrôle une expérience. Il est possible de l’éditer directement ou d’en copier une pour garder des réglages distincts :

~~~powershell
Copy-Item config/config.json config/config-extra-trees.json
~~~

Voir configuration.md pour les différentes clés et valeurs acceptés.

## 5. Entraîner et évaluer sur validation

Entraîner un modèle et le valider :

~~~powershell
.\.venv\Scripts\python.exe -m src.train --config config/nom_config.json
~~~

Pour rechercher les hyperparamètres décrits dans `hyperparameter_search` :

~~~powershell
.\.venv\Scripts\python.exe -m src.hyperparameter_search --config config/nom_config.json
~~~

Pour créer une soumission :

~~~powershell
.\.venv\Scripts\python.exe -m src.predict --config config/nom_config.json
~~~


Chaque exécution écrit un sous-dossier horodaté sous artifacts/. En mode holdout, il contient la configuration, validation_score, les prédictions et le modèle ajusté sur la partition d’entraînement. En mode stratified_kfold, metrics.json contient le ROC-AUC moyen dans validation_score et chaque score dans fold_validation_scores, aucun modèle ni fichier de prédictions par pli n’est enregistré.
Seuls quelques deux exemples sont conservés sur github, car les fichiers sont lourds (1Go au total).

