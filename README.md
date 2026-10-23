# Challenge de classification des demandes de prêt immobilier

Ce dépôt contient le code reproductible pour entraîner un classifieur et produire une soumission Kaggle. La cible est **accepted**; le score utilisé pour comparer les expériences est le **ROC-AUC**. Les fichiers de données ne doivent pas être ajoutés au dépôt.

## 1. Préparer l'environnement

Les dépendances sont fixées dans pyproject.toml et uv.lock. Avec Python 3.12 sous Windows PowerShell, installer uv une fois, puis synchroniser l'environnement depuis la racine du dépôt :

~~~powershell
python -m pip install uv
python -m uv sync --locked
~~~

Sous macOS ou Linux, installer uv une fois puis synchroniser de la même façon :

~~~bash
python3 -m pip install uv
python3 -m uv sync --locked
~~~

## 2. Placer les données localement

Le programme attend les fichiers **train.csv**, **test.csv** et **sample_submission.csv** dans le dossier **data/**. Ce dossier est ignoré par Git. Si les fichiers se trouvent ailleurs, modifier **data_dir** dans config.json, avec un chemin relatif à la racine du projet ou un chemin absolu. Le lecteur préserve les catégories textuelles littérales comme NA et interprète les marqueurs de valeurs manquantes des variables numériques.

Ne pas utiliser **git add -f** sur les CSV. Pour confirmer que Git les ignore :

~~~powershell
git check-ignore -v data/train.csv data/test.csv data/sample_submission.csv
~~~

## 3. Choisir une expérience

Modifier config.json avant chaque expérience :

- **model** : logistic_regression, random_forest, extra_trees ou xgboost.
- **model_params** : paramètres du modèle sélectionné. Pour changer de modèle, remplacer aussi cet objet par des paramètres compatibles. Les valeurs par défaut des modèles sont définies dans src/models.py.
- **master_seed** : graine passée au découpage et aux modèles stochastiques.
- **validation_size** : proportion gardée pour validation (défaut : 20 %, stratifié).
- **preprocessing.scale_numeric** : active ou désactive la standardisation des variables numériques.
- **preprocessing.add_missing_indicators** : ajoute ou non des indicateurs pour les valeurs manquantes.
- **preprocessing.log1p_columns** : liste des variables numériques non négatives auxquelles appliquer log(1+x), par exemple ["loan_amount", "income"].

L’imputation et l’encodage des catégories sont obligatoires pour les modèles pris en charge. Les codes numériques, les variables géographiques, les tranches d’âge et debt_to_income_ratio sont traités comme des catégories. id est conservé pour la soumission, jamais comme variable explicative.

## 4. Entraîner et évaluer

Depuis la racine du dépôt :

~~~powershell
.\.venv\Scripts\python.exe -m src.train --config config.json
~~~

Le programme sépare train.csv en sous-ensembles d'entraînement et de validation, en conservant la proportion des deux classes. Le prétraitement est ajusté uniquement sur le sous-ensemble d'entraînement. Le ROC-AUC est calculé sur les probabilités de accepted = 1, pas sur des prédictions binaires.

Chaque exécution crée un sous-dossier dans artifacts/ contenant la configuration utilisée, le modèle ajusté sur la partition d'entraînement, le ROC-AUC, les prédictions de validation et des informations de reproductibilité. Comparer les scores après avoir modifié config.json. Réutiliser la même validation pour de nombreux réglages peut conduire à un score optimiste.

## 5. Produire une soumission Kaggle

Après avoir choisi le modèle et ses paramètres :

~~~powershell
.\.venv\Scripts\python.exe -m src.predict --config config.json
~~~

Cette commande réentraîne le pipeline sur tout train.csv, prédit des probabilités pour test.csv, puis écrit un CSV conforme à sample_submission.csv dans submissions/. Le modèle final et les métadonnées de l'exécution sont enregistrés dans artifacts/.

## 6. Explorer les données et suivre le tutoriel

- Notebook d'exploration : [notebooks/exploration_donnees.ipynb](notebooks/exploration_donnees.ipynb). Il présente les distributions, valeurs manquantes et statistiques descriptives du jeu d'entraînement.
- Guide pas à pas (environnement, configuration, validation, comparaison et soumission) : [docs/guide_execution.md](docs/guide_execution.md).

Pour ouvrir le notebook après l'installation des dépendances :

~~~powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/exploration_donnees.ipynb
~~~

## Organisation

- src/data.py : lecture, contrôles du schéma et découpage stratifié.
- src/preprocessing.py : imputation, encodage et transformations optionnelles.
- src/models.py : registre des modèles et pipelines scikit-learn.
- src/train.py : entraînement et évaluation sur validation.
- src/predict.py : réentraînement complet et création de la soumission.
- src/hyperparameter_search.py : emplacement réservé; la recherche automatique n'est pas encore implémentée.
- notebooks/exploration_donnees.ipynb : exploration descriptive du train.
- docs/guide_execution.md : procédure d'utilisation détaillée.

Les dépendances, les données, les modèles entraînés et les soumissions générées sont exclus du dépôt par .gitignore lorsqu'ils sont dans leurs dossiers respectifs.
