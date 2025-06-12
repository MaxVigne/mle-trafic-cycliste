from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from streamlit_utils.streamlit_utils import load_classification_data, train_classification_model
import joblib

MODELS_DIR = Path("models")

class_df = load_classification_data()
X = class_df.drop(columns=["Comptage horaire"])
col_norm = ["Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]

# Encodage des features
encoder = OneHotEncoder(sparse_output=False, dtype=int)
array = encoder.fit_transform(X[col_norm])
encoded_df_clean = pd.DataFrame(array, columns=encoder.get_feature_names_out(col_norm))
encoded_df_clean.index = X.index
X_clean = pd.concat([X.drop(columns=col_norm), encoded_df_clean], axis=1)

# Encodage variable cible
label_enc = LabelEncoder()
y = class_df["Comptage horaire"]
y = label_enc.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

n_estimators = [200]
max_depth = [100]

for n_est in n_estimators:
    for md in max_depth:
        params = {'n_estimators': n_est, 'max_depth': md, 'criterion': 'gini', 'min_samples_split':15, 'min_samples_leaf':2, 'max_features':'sqrt'}
        print("training with")
        print(params)
        # On génère un nom de fichier unique qui est basé sur les hyperparamètres
        model_filename = MODELS_DIR / f"rf_classifier_{n_est}_{md}.pkl"

        # Ici on vérifie si le modèle est déjà entraîné
        if model_filename.exists():
            print("Chargement du modèle pré-entraîné...")
            model = joblib.load(model_filename)
        else:
            print("Entraînement du modèle... (peut prendre quelques minutes)")
            model = train_classification_model(X_train, y_train, params)
            joblib.dump(model, model_filename)
            print("Modèle entraîné et sauvegardé pour une utilisation future!")
