# App_sup_M2

# Prédiction de performance en Powerlifting (`TotalKg`)

Modélisation de régression sur un échantillon du dataset **OpenPowerlifting** afin de prédire le total réalisé (`TotalKg` = squat + développé couché + soulevé de terre).  
La métrique d'évaluation du challenge est la **RMSE** (*Root Mean Squared Error*).

---

## Arborescence du projet

```text
.
├── README.md
└── regression/
    ├── data/                  # (Ignoré par Git) Données brutes train.csv et test.csv
    ├── results/               # (Ignoré par Git) Fichiers de soumission générés (.csv)
    ├── linear_regression.py   # Baseline naïve (OneHotEncoder + StandardScaler + LinearRegression)
    ├── lightGBM.py            # Validation croisée 5-Folds + sélection combinatoire de paquets de features
    └── lightGBM_optuna.py     # Optimisation bayésienne des hyperparamètres (Optuna) + entraînement final
```

> **Note sur `.gitignore` :** Les dossiers `regression/data/` et `regression/results/` sont ignorés par Git. Avant de lancer les scripts, assurez-vous de créer ces deux dossiers en local et de placer `train.csv` et `test.csv` dans `regression/data/`.

---

## Installation des dépendances

Assurez-vous d'utiliser Python 3.9+ et d'installer les bibliothèques requises :

```bash
pip install pandas numpy scikit-learn lightgbm optuna
```

---

## Exécution des scripts

Tous les scripts doivent être lancés depuis la **racine du dépôt** (car les chemins relatifs pointent vers `regression/data/` et `regression/results/`).

1. **Modèle de référence (Baseline linéaire) :**
   ```bash
   python regression/linear_regression.py
   ```
   *Génère `regression/results/submission_naive.csv`.*

2. **Sélection des paquets de features avec LightGBM (5-Fold CV) :**
   ```bash
   python regression/lightGBM.py
   ```
   *Teste toutes les combinaisons de paquets (`Geo`, `Physics`, `Freq`), affiche l'importance des variables (Gain) et génère un fichier de soumission par combinaison dans `regression/results/`.*

3. **Optimisation des hyperparamètres avec Optuna :**
   ```bash
   python regression/lightGBM_optuna.py
   ```
   *Lance la recherche d'hyperparamètres sur 5 folds et génère `regression/results/submission_lgbm_optuna.csv`.*

---

## Configuration des paquets de features

En plus des **features de base** (`Is_Home_Country`, `Sex_Equipment`, `WeightClass_Num`, `Weight_Diff`, `Weight_Age_Ratio`), trois paquets additionnels ont été conçus :
* **`Geo`** : `Is_Home_State`, `Travel_Type` *(meilleur compromis biais/variance en validation locale)*.
* **`Physics`** : `Pseudo_BMI`, `Weight_Year_Ratio`.
* **`Freq`** : `Fed_Popularity`, `Country_Popularity`.

### Modifier les paquets dans `lightGBM.py` (sans tuning Optuna)
Dans `regression/lightGBM.py`, modifiez la liste `packs_disponibles` pour choisir les paquets à évaluer combinatoirement :

```python
# Exemple pour tester uniquement Geo et Freq :
packs_disponibles = ['Geo', 'Freq']
```

### Modifier les paquets dans `lightGBM_optuna.py` (avec tuning Optuna)
Dans `regression/lightGBM_optuna.py`, commentez ou décommentez les blocs souhaités dans la boucle `for df in [train, test]:` :

```python
for df in [train, test]:
    # Base (toujours conservé)
    df['Is_Home_Country'] = (df['Country'] == df['MeetCountry']).astype(str)
    df['Sex_Equipment'] = df['Sex'].astype(str) + "_" + df['Equipment'].astype(str)
    df['WeightClass_Num'] = df['WeightClassKg'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    df['Weight_Diff'] = df['WeightClass_Num'] - df['BodyweightKg']
    df['Weight_Age_Ratio'] = df['BodyweightKg'] / (df['Age'] + 1)
    
    # Paquet Geo (recommandé : minimise la RMSE locale et publique)
    df['Is_Home_State'] = (df['State'] == df['MeetState']).astype(str)
    df['Travel_Type'] = df['Is_Home_Country'] + "_" + df['Is_Home_State']
    
    # Paquet Physics (commenter les 2 lignes ci-dessous pour désactiver)
    # df['Pseudo_BMI'] = df['BodyweightKg'] / ((df['WeightClass_Num'] + 10) ** 2)
    # df['Weight_Year_Ratio'] = df['BodyweightKg'] / (df['MeetYear'] - 1970)
    
    # Paquet Freq (commenter les 4 lignes ci-dessous pour désactiver)
    # freq_fed = df['Federation'].value_counts()
    # df['Fed_Popularity'] = df['Federation'].map(freq_fed)
    # freq_country = df['Country'].value_counts()
    # df['Country_Popularity'] = df['Country'].map(freq_country)
```

> **Attention :** Si vous désactivez le paquet `Geo`, pensez également à retirer `'Is_Home_State'` et `'Travel_Type'` de la liste `categorical_features` juste en dessous, sinon Pandas lèvera une erreur `KeyError`.

---

## Suivi des performances (RMSE)

| Itération / Modèle | RMSE Locale (OOF CV) | Leaderboard Public (30%) |
| :--- | :---: | :---: |
| Régression Linéaire (Baseline) | 85.98 kg | 86.907 kg |
| LightGBM (Split simple, features brutes) | 78.52 kg | 78.869 kg |
| LightGBM (5-Fold CV, features brutes) | 78.57 kg | 78.627 kg |
| LightGBM (5-Fold CV + Features Base) | 78.40 kg | 78.436 kg |
| LightGBM (5-Fold CV + Base + `Geo`) | 78.345 kg | 78.345 kg |
| **LightGBM + Optuna (Base + `Geo`)** | **77.756 kg** | **77.923 kg** |
| LightGBM + Optuna (Base + `Geo` + `Physics` + `Freq`) | 77.739 kg | 78.051 kg |