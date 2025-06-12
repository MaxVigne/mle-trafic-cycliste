import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from pathlib import Path
import joblib

# MLflow tracking sur DagsHub (need MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD to be defined)
mlflow.set_tracking_uri("https://dagshub.com/MaxVigne/mle-trafic-cycliste.mlflow")
mlflow.set_experiment("modele_trafic_cycliste")

reg_df = pd.read_csv('data/processed/lieu-compteur-one-hot-encoded.csv', index_col=0)
X = reg_df.drop(columns=["Comptage horaire"])

# Transformation logarithmique de la variable cible pour éviter les valeurs aberrantes
y = np.log1p(reg_df["Comptage horaire"])  # log(1+x) pour gérer les zéros

learning_rate = 0.5
max_iter = 500

with mlflow.start_run(run_name=f"hgb_regressor_lr_{learning_rate}_iter_{max_iter}") as run:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {'learning_rate': learning_rate, 'max_iter': max_iter}
    mlflow.log_params(params)

    MODELS_DIR = Path("models")

    model_filename = MODELS_DIR / f"hgb_regressor_{learning_rate}_{max_iter}.pkl"
    model = HistGradientBoostingRegressor(**params)
    print(f"Training du modèle avec les paramètres {params}")
    model.fit(X_train, y_train)
    print(f"Sauvegarde du modèle dans {model_filename}")
    joblib.dump(model, model_filename)

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Scores du modèles sur l'ensemble de test : RMSE={rmse:.4f} et R²={r2:.4f}")

    print(f"Sauvegarde des scores sur MLFLow")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
