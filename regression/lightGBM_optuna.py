import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# chargement et prépa des données
train = pd.read_csv('regression/data/train.csv')
test = pd.read_csv('regression/data/test.csv')

target = 'TotalKg'
id_col = 'id'

for df in [train, test]:
    # base
    df['Is_Home_Country'] = (df['Country'] == df['MeetCountry']).astype(str)
    df['Sex_Equipment'] = df['Sex'].astype(str) + "_" + df['Equipment'].astype(str)
    df['WeightClass_Num'] = df['WeightClassKg'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    df['Weight_Diff'] = df['WeightClass_Num'] - df['BodyweightKg']
    df['Weight_Age_Ratio'] = df['BodyweightKg'] / (df['Age'] + 1)
    
    # paquet de features geo (paquet ayant minimisé la rmse locale)
    df['Is_Home_State'] = (df['State'] == df['MeetState']).astype(str)
    df['Travel_Type'] = df['Is_Home_Country'] + "_" + df['Is_Home_State']

categorical_features = [
    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg', 
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation', 
    'MeetCountry', 'MeetState', 'Sanctioned', 
    'Is_Home_Country', 'Sex_Equipment', 'Is_Home_State', 'Travel_Type'
]

for col in categorical_features:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# objectif optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
        'n_estimators': 2000,
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_X=X_va, eval_y=y_va,
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_va)
        
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    return rmse

# lancement opti
print("=== DÉBUT DE L'OPTIMISATION OPTUNA ===")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("\n========================================")
print(f"Meilleure RMSE trouvée : {study.best_value:.3f} kg")
print("Meilleurs hyperparamètres :")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print("========================================\n")

# entraînement final & submission
print("Entraînement du modèle final (CV 5-Folds) avec les paramètres optimisés...")
best_params = study.best_params
best_params.update({
    'objective': 'regression', 'metric': 'rmse', 
    'random_state': 42, 'n_jobs': -1, 'verbose': -1, 'n_estimators': 2000
})

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_X=X_va, eval_y=y_va,
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    
    # moyenne des 5 folds
    test_preds += model.predict(X_test) / 5

submission = pd.DataFrame({
    id_col: test[id_col],
    target: test_preds
})
submission.to_csv('regression/results/submission_lgbm_optuna.csv', index=False)
print("Fichier de soumission 'submission_lgbm_optuna.csv' généré avec succès !")