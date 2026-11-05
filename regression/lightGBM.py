import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# 1. Chargement des données
train = pd.read_csv('regression/data/train.csv')
test = pd.read_csv('regression/data/test.csv')

target = 'TotalKg'
id_col = 'id'

# Préparation des features
X = train.drop(columns=[target, id_col])
y = train[target]
X_test_submission = test.drop(columns=[id_col])

# Variables catégorielles
categorical_features = [
    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg', 
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation', 
    'MeetCountry', 'MeetState', 'Sanctioned'
]

# Conversion en type "category"
for col in categorical_features:
    X[col] = X[col].astype('category')
    X_test_submission[col] = X_test_submission[col].astype('category')

# Configuration de la validation croisée (5 folds)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Tableaux pour stocker les prédictions
oof_predictions = np.zeros(len(X)) # Out-Of-Fold : prédictions de validation
test_predictions = np.zeros(len(X_test_submission)) # Prédictions de test (somme)

# Boucle sur chaque fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Entraînement du Fold {fold + 1}/{n_splits} ---")
    
    # Séparation Train/Val pour ce fold spécifique
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialisation du modèle
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    # Entraînement avec Early Stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # Masque les logs à chaque arbre pour un affichage propre
        ]
    )
    
    # prédiction sur la validation (OOF)
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"RMSE du Fold {fold + 1} : {fold_rmse:.2f} kg (Meilleure itération : {model.best_iteration_})")
    
    # prédiction sur le set de test (Accumulation pour faire la moyenne)
    # LightGBM utilise automatiquement best_iteration_ lors de l'appel à predict()
    test_predictions += model.predict(X_test_submission) / n_splits

# évaluation globale (OOF - Out Of Fold)
global_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\n========================================")
print(f"RMSE locale globale (OOF CV) : {global_rmse:.2f} kg")
print(f"========================================")

# création du fichier de soumission
submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_predictions
})

submission.to_csv('regression/results/submission_lgbm_cv.csv', index=False)
print("Fichier de soumission 'submission_lgbm_cv.csv' généré avec succès !")