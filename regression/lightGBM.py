import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# fonction de Feature Engineering
def create_features(df):
    df = df.copy()
    
    # avantage à domicile
    df['Is_Home_Country'] = (df['Country'] == df['MeetCountry']).astype(str)
    
    # interaction Sexe et Équipement : l'impact du matériel (ex: maillot de force) varie énormément selon le sexe
    df['Sex_Equipment'] = df['Sex'].astype(str) + "_" + df['Equipment'].astype(str)
    
    # poids brut vs Catégorie de poids
    # Extraction de la valeur numérique de la catégorie (ex: "93+" -> 93.0, "83" -> 83.0)
    df['WeightClass_Num'] = df['WeightClassKg'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    
    # différence de poids : indique si l'athlète a fait un "cutting" extrême ou s'il est léger pour sa catégorie
    df['Weight_Diff'] = df['WeightClass_Num'] - df['BodyweightKg']
    
    # indicateur composite de densité corporelle et d'âge
    df['Weight_Age_Ratio'] = df['BodyweightKg'] / (df['Age'] + 1) # +1 pour éviter division par 0
    
    return df

# chargement des données
train = pd.read_csv('regression/data/train.csv')
test = pd.read_csv('regression/data/test.csv')

target = 'TotalKg'
id_col = 'id'

# application du Feature Engineering avant séparation
train = create_features(train)
test = create_features(test)

X = train.drop(columns=[target, id_col])
y = train[target]
X_test_submission = test.drop(columns=[id_col])

# variables catégorielles
categorical_features = [
    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg', 
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation', 
    'MeetCountry', 'MeetState', 'Sanctioned', 
    'Is_Home_Country', 'Sex_Equipment' # Nouvelles features catégorielles
]

# Conversion en type "category"
for col in categorical_features:
    X[col] = X[col].astype('category')
    X_test_submission[col] = X_test_submission[col].astype('category')

# validation croisée
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test_submission))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Entraînement du Fold {fold + 1}/{n_splits} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    # utilisation de eval_X et eval_y pour éviter le warning de dépréciation
    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"RMSE du Fold {fold + 1} : {fold_rmse:.2f} kg (Meilleurs arbres : {model.best_iteration_})")
    
    test_predictions += model.predict(X_test_submission) / n_splits

# évaluation globale
global_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\n========================================")
print(f"RMSE locale globale (OOF CV) avec Feature Engineering : {global_rmse:.2f} kg")
print(f"========================================")

submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_predictions
})
submission.to_csv('regression/results/submission_lgbm_features.csv', index=False)
print("Fichier de soumission 'submission_lgbm_features.csv' généré avec succès !")