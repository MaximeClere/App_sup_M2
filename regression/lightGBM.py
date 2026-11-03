import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# Chargement des données
train = pd.read_csv('regression/data/train.csv')
test = pd.read_csv('regression/data/test.csv')

target = 'TotalKg'
id_col = 'id'

# Préparation des features
X = train.drop(columns=[target, id_col])
y = train[target]
X_test_submission = test.drop(columns=[id_col])

# Liste des variables catégorielles
categorical_features = [
    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg', 
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation', 
    'MeetCountry', 'MeetState', 'Sanctioned'
]

# Conversion explicite au type "category" (obligatoire pour LightGBM)
for col in categorical_features:
    X[col] = X[col].astype('category')
    X_test_submission[col] = X_test_submission[col].astype('category')

# Validation locale (Train/Val split) pour estimer la RMSE et trouver le bon nombre d'arbres
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle avec un grand nombre d'arbres max, l'early stopping l'arrêtera au bon moment
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

print("Entraînement du modèle de validation avec Early Stopping...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

y_val_pred = model.predict(X_val)

# Calcul de la RMSE locale
local_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"RMSE de validation locale (LightGBM) : {local_rmse:.2f} kg")

# Entraînement sur l'ensemble des données train.csv pour la soumission
print("Entraînement final sur l'ensemble des données...")
# On récupère le nombre d'arbres optimal trouvé lors de l'étape précédente
optimal_trees = model.best_iteration_

final_model = lgb.LGBMRegressor(
    n_estimators=optimal_trees,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)

# Prédiction sur le set de test
test_predictions = final_model.predict(X_test_submission)

# Création du fichier de soumission
submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_predictions
})

# Sauvegarde au format attendu (dans le dossier results)
submission.to_csv('regression/results/submission_lgbm.csv', index=False)
print("Fichier de soumission 'submission_lgbm.csv' généré avec succès !")