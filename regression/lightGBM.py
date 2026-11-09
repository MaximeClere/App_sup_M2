import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from itertools import combinations
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def get_base_data():
    train = pd.read_csv('regression/data/train.csv')
    test = pd.read_csv('regression/data/test.csv')
    
    # Features de base + celles validées à l'étape précédente
    for df in [train, test]:
        df['Is_Home_Country'] = (df['Country'] == df['MeetCountry']).astype(str)
        df['Sex_Equipment'] = df['Sex'].astype(str) + "_" + df['Equipment'].astype(str)
        df['WeightClass_Num'] = df['WeightClassKg'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        df['Weight_Diff'] = df['WeightClass_Num'] - df['BodyweightKg']
        df['Weight_Age_Ratio'] = df['BodyweightKg'] / (df['Age'] + 1)
        
    return train, test

# Définition des "Paquets" de features
def apply_feature_packs(df, packs_to_apply):
    df = df.copy()
    new_cat_features = []
    
    # paqueet 1 : Variables Géographiques locales
    if 'Geo' in packs_to_apply:
        df['Is_Home_State'] = (df['State'] == df['MeetState']).astype(str)
        df['Travel_Type'] = df['Is_Home_Country'] + "_" + df['Is_Home_State']
        new_cat_features.extend(['Is_Home_State', 'Travel_Type'])
        
    # paquet 2 : Physiques et Interactions
    if 'Physics' in packs_to_apply:
        # Proxy d'IMC (si on considère WeightClass comme un proxy de carrure)
        df['Pseudo_BMI'] = df['BodyweightKg'] / ((df['WeightClass_Num'] + 10) ** 2)
        # Ratio Poids / Année de compétition (évolution temporelle des physiques)
        df['Weight_Year_Ratio'] = df['BodyweightKg'] / (df['MeetYear'] - 1970)
        
    # paquet 3 : Fréquences d'apparition (Popularité)
    if 'Freq' in packs_to_apply:
        # Permet au modèle de différencier une fédération mondiale d'une fédération locale
        freq_fed = df['Federation'].value_counts()
        df['Fed_Popularity'] = df['Federation'].map(freq_fed)
        freq_country = df['Country'].value_counts()
        df['Country_Popularity'] = df['Country'].map(freq_country)
        
    return df, new_cat_features

train_base, test_base = get_base_data()
target = 'TotalKg'
id_col = 'id'

base_cat_features = [
    'Sex', 'Equipment', 'AgeClass', 'BirthYearClass', 'WeightClassKg', 
    'Tested', 'Country', 'State', 'Federation', 'ParentFederation', 
    'MeetCountry', 'MeetState', 'Sanctioned', 'Is_Home_Country', 'Sex_Equipment'
]

# tester toutes les combinaisons de paquets
packs_disponibles = ['Geo', 'Physics', 'Freq']
toutes_combinaisons = []
for i in range(1, len(packs_disponibles) + 1):
    toutes_combinaisons.extend(combinations(packs_disponibles, i))

best_rmse = float('inf')
best_comb = None
best_model = None # Pour extraire l'importance plus tard

print("=== DEBUT DU GRID SEARCH DE FEATURES ===")

for combo in toutes_combinaisons:
    combo_name = "+".join(combo)
    print(f"\n--- Test de la combinaison : {combo_name} ---")
    
    # application des features
    train_df, new_cats = apply_feature_packs(train_base, combo)
    test_df, _ = apply_feature_packs(test_base, combo)
    
    cat_cols = base_cat_features + new_cats
    for col in cat_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')
        
    X = train_df.drop(columns=[target, id_col])
    y = train_df[target]
    X_test = test_df.drop(columns=[id_col])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # on stocke l'importance des features du dernier fold
    feature_importances = np.zeros(X.shape[1])
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1)
        model.fit(
            X_tr, y_tr, eval_X=X_va, eval_y=y_va,
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
        )
        
        oof_preds[val_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / 5
        # Gain d'information accumulé par feature
        feature_importances += model.booster_.feature_importance(importance_type='gain') / 5
        
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f">> RMSE (OOF) pour {combo_name} : {rmse:.3f} kg")
    
    # sauvegarde de la soumission pour cette combinaison
    pd.DataFrame({id_col: test_base[id_col], target: test_preds}).to_csv(f'regression/results/sub_{combo_name}.csv', index=False)
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_comb = combo_name
        best_importances = feature_importances
        best_features = X.columns

print(f"\n========================================")
print(f"Meilleure combinaison : {best_comb} avec {best_rmse:.3f} kg")
print(f"========================================\n")

# Affichage du Top 15 Feature Importance (basé sur le Gain)
importance_df = pd.DataFrame({'Feature': best_features, 'Gain': best_importances})
importance_df = importance_df.sort_values(by='Gain', ascending=False).head(15)
print("--- TOP 15 FEATURE IMPORTANCE (GAIN) ---")
print(importance_df.to_string(index=False))