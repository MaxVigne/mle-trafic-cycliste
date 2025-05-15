import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

reg_df = pd.read_csv('data/processed/lieu-compteur-one-hot-encoded.csv', index_col=0)
X = reg_df.drop(columns=["Comptage horaire"])

# Transformation logarithmique de la variable cible pour éviter les valeurs aberrantes
y = np.log1p(reg_df["Comptage horaire"])  # log(1+x) pour gérer les zéros

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

learning_rate = 0.5
max_iter = 500
params = {'learning_rate': learning_rate, 'max_iter': max_iter}
MODELS_DIR = Path("models")

model_filename = MODELS_DIR / f"hgb_regressor_{learning_rate}_{max_iter}.pkl"
model = HistGradientBoostingRegressor(**params)
print(f"Training du modèle avec les paramètres {params}")
model.fit(X_train, y_train)
print(f"Sauvegarde du modèle dans {model_filename}")
joblib.dump(model, model_filename)

# TODO Calcul des scores et enregistrement dans MLFlow
