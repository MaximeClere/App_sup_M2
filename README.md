# Classification des demandes de prêt immobilier

Ce dépôt contient le code du Challenge de classification binaire des demandes de prêt immobilier. Le code est organisé autour de la lecture des données, du prétraitement, de l'entraînement et de l'évaluation des modèles, puis de la génération des soumissions.

## Structure du code

```text
src/
├── config.py                 # Lecture et validation des configurations JSON
├── data.py                   # Chargement des données et découpages reproductibles
├── preprocessing.py          # Prétraitement tabulaire intégré aux pipelines
├── models.py                 # Registre des modèles et construction des pipelines
├── train.py                  # Entraînement et évaluation sur validation
├── hyperparameter_search.py  # Recherche par grille ou aléatoire
├── predict.py                # Réentraînement complet et prédiction du jeu test
└── artifacts.py              # Enregistrement des modèles et résultats

notebooks/
├── exploration_donnees.ipynb       # Exploration descriptive du jeu de données
└── data_engineering_config_1.ipynb # Étude des transformations de variables

docs/
├── guide_execution.md  # Préparation, configurations et utilisation du projet
└── configuration.md    # Référence des paramètres des fichiers JSON
```

Les configurations JSON à la racine du dossier `config/` décrivent les expériences et sont chargées par `src/config.py`. Les pipelines sont construits par `src/models.py` en associant le prétraitement à l'estimateur choisi. `src/train.py` et `src/hyperparameter_search.py` évaluent les modèles, `src/predict.py` produit les prédictions destinées à une soumission et `src/artifacts.py` rassemble les sorties des exécutions.

## Documentation

- [Guide d'exécution](docs/guide_execution.md) : installation, données, configurations et commandes du projet.
- [Référence de configuration](docs/configuration.md) : clés JSON, valeurs acceptées et paramètres des méthodes.
