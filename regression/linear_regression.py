import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Chargement des données
# Assurez-vous que les fichiers CSV sont dans le même dossier que le script
train = pd.read_csv('regression/data/train.csv')
test = pd.read_csv('regression/data/test.csv')

# 2. Préparation des variables
target = 'TotalKg'
id_col = 'id'

# Séparation des features (X) et de la cible (y)
X = train.drop(columns=[target, id_col])
y = train[target]

# Liste des variables numériques et catégorielles selon la description
numeric_features = ['Age', 'BodyweightKg', 'MeetYear']
categorical_features = [
    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg', 
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation', 
    'MeetCountry', 'MeetState', 'Sanctioned'
]

# 3. Création du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        # handle_unknown='ignore' évite les erreurs si une nouvelle catégorie apparaît dans test.csv
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Création du Pipeline avec la Régression Linéaire
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Validation locale (Train/Test split) pour estimer la RMSE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)
y_test_pred = model_pipeline.predict(X_test)

# Calcul de la RMSE locale
# On utilise np.sqrt avec mean_squared_error pour une compatibilité maximale avec toutes les versions de scikit-learn
local_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"RMSE de validation locale : {local_rmse:.2f} kg")

# 6. Entraînement sur l'ensemble des données d'entraînement pour la soumission
print("Entraînement sur l'ensemble des données train.csv...")
model_pipeline.fit(X, y)

# 7. Prédiction sur le set de test
X_test = test.drop(columns=[id_col])
test_predictions = model_pipeline.predict(X_test)

# 8. Création du fichier de soumission
submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_predictions
})

# Sauvegarde au format attendu (sans l'index pandas)
submission.to_csv('regression/results/submission_linreg.csv', index=False)
print("Fichier de soumission 'submission_naive_linreg.csv' généré avec succès !")