from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from pathlib import Path

app = FastAPI(title="Traffic Cycliste Prediction API",
              description="API for predicting bicycle traffic in Paris",
              version="1.0.0")

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "hgb_regressor_0.5_500.pkl"
ENCODER_PATH = MODELS_DIR / "one_hot_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"Error loading model and/or one hot encoder: {e}")
    raise e

class ModelInputParameters(BaseModel):
    jour: int
    mois: int
    annee: int
    heure: int
    jour_semaine: int
    jour_ferie: bool
    vacances_scolaires: bool
    site_comptage: str

class PredictionResponse(BaseModel):
    prediction: float
    input_parameters: ModelInputParameters

@app.get("/")
def index():
    return {"message": "Welcome to the Traffic Cycliste Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_parameters: ModelInputParameters):
    try:
        # Transform input into format expected by model
        x = pd.DataFrame([{
            "Nom du site de comptage": input_parameters.site_comptage,
            "Jour": input_parameters.jour,
            "Mois": input_parameters.mois,
            "Année": input_parameters.annee,
            "Heure": input_parameters.heure,
            "Jour_semaine": input_parameters.jour_semaine,
            "Jour férié": int(input_parameters.jour_ferie),
            "Vacances scolaires": int(input_parameters.vacances_scolaires),
        }])

        # One Hot Encoding
        X_p = encoder.transform(x)

        # Prediction
        y_p = np.expm1(model.predict(X_p))

        return PredictionResponse(
            prediction=y_p,
            input_parameters=input_parameters
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
