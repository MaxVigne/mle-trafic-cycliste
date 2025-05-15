import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

# On crée le dataset processed/lieux-compteur-one-hot-encoded.csv
# utilisé pour l'entrainement de notre modèle de régression.

# Import des datasets
df_2023 = pd.read_csv('data/raw/velo_2023.csv', sep=';')
df_2024 = pd.read_csv('data/raw/velo_2024.csv', sep=';')

# Concaténation des années 2023 et 2024
df = pd.concat([df_2023, df_2024], axis=0)

# Ajout des données manquantes pour certains compteurs
DATA_COMPTEURS_MANQUANTS = {
    'Face au 48 quai de la marne NE-SO': {
        'Identifiant du compteur': '100047542-103047542',
        'Identifiant du site de comptage': 100047542,
        'Nom du site de comptage': 'Face au 48 quai de la marne',
        "Date d'installation du site de comptage": '2018-11-29',
        'Coordonnées géographiques': '48.89128,2.38606',
    },
    'Face au 48 quai de la marne SO-NE': {
        'Identifiant du compteur': '100047542-103047542',
        'Identifiant du site de comptage': 100047542,
        'Nom du site de comptage': 'Face au 48 quai de la marne',
        "Date d'installation du site de comptage": '2018-11-29',
        'Coordonnées géographiques': '48.89128,2.38606',
    },
    # start
    'Quai des Tuileries NO-SE': {
        'Identifiant du compteur': '100056035-353266462',
        'Identifiant du site de comptage': 100056035,
        'Nom du site de comptage': 'Quai des Tuileries',
        "Date d'installation du site de comptage": '2021-05-18',
        'Coordonnées géographiques': '48.8635,2.32239',
    },
    'Quai des Tuileries SE-NO': {
        'Identifiant du compteur': '100056035-353266462',
        'Identifiant du site de comptage': 100056035,
        'Nom du site de comptage': 'Quai des Tuileries',
        "Date d'installation du site de comptage": '2021-05-18',
        'Coordonnées géographiques': '48.8635,2.32239',
    },
    'Pont des Invalides N-S': {
        'Identifiant du compteur': '100056223-101056223',
        'Identifiant du site de comptage': 100056223,
        'Nom du site de comptage': 'Pont des Invalides',
        "Date d'installation du site de comptage": '2019-11-07',
        'Coordonnées géographiques': '48.86281,2.31037',
    },
    '10 avenue de la Grande Armée SE-NO': {
        'Identifiant du compteur': '100044494-353504987',
        'Identifiant du site de comptage': 100044494,
        'Nom du site de comptage': '10 avenue de la Grande Armée',
        "Date d'installation du site de comptage": '2018-07-27',
        'Coordonnées géographiques': '48.8748,2.2924',
    },
    '27 quai de la Tournelle NO-SE': {
        'Identifiant du compteur': '100056336-104056336',
        'Identifiant du site de comptage': 100056336,
        'Nom du site de comptage': '27 quai de la Tournelle',
        "Date d'installation du site de comptage": '2019-11-14',
        'Coordonnées géographiques': '48.85013,2.35423',
    },
    '27 quai de la Tournelle SE-NO': {
        'Identifiant du compteur': '100056336-104056336',
        'Identifiant du site de comptage': 100056336,
        'Nom du site de comptage': '27 quai de la Tournelle',
        "Date d'installation du site de comptage": '2019-11-14',
        'Coordonnées géographiques': '48.85013,2.35423',
    },
}

for compteur in DATA_COMPTEURS_MANQUANTS:
    cond = (df['Nom du compteur'] == compteur) & (df['Identifiant du compteur'].isna())
    df.loc[cond, 'Identifiant du compteur'] = DATA_COMPTEURS_MANQUANTS[compteur]['Identifiant du compteur']
    df.loc[cond, 'Identifiant du site de comptage'] = DATA_COMPTEURS_MANQUANTS[compteur]['Identifiant du site de comptage']
    df.loc[cond, 'Nom du site de comptage'] = DATA_COMPTEURS_MANQUANTS[compteur]['Nom du site de comptage']
    df.loc[cond, "Date d'installation du site de comptage"] = DATA_COMPTEURS_MANQUANTS[compteur]["Date d'installation du site de comptage"]
    df.loc[cond, 'Coordonnées géographiques'] = DATA_COMPTEURS_MANQUANTS[compteur]['Coordonnées géographiques']

# Suppression des variables inutiles pour la modélisation
drop_columns = columns = [
    "Identifiant technique compteur",
    "ID Photos",
    "test_lien_vers_photos_du_site_de_comptage_",
    "id_photo_1",
    "url_sites",
    "type_dimage",
    "mois_annee_comptage",
    "Lien vers photo du site de comptage",
    "Date d'installation du site de comptage",
    "Identifiant du compteur",
    "Identifiant du site de comptage",
    "Coordonnées géographiques"
]

df_clean = df.drop(columns = drop_columns)

# Fusion des compteurs doublons identifiés pendant l'exploration
df_clean = df_clean.replace({
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos NE-SO': 'Face au 48 quai de la marne NE-SO',
    'Face au 48 quai de la marne Face au 48 quai de la marne Vélos SO-NE': 'Face au 48 quai de la marne SO-NE',
    'Totem 64 Rue de Rivoli Totem 64 Rue de Rivoli Vélos E-O': 'Totem 64 Rue de Rivoli E-O',
    'Totem 64 Rue de Rivoli Totem 64 Rue de Rivoli Vélos O-E': 'Totem 64 Rue de Rivoli O-E',
    'Quai des Tuileries Quai des Tuileries Vélos NO-SE': 'Quai des Tuileries NO-SE',
    'Quai des Tuileries Quai des Tuileries Vélos SE-NO': 'Quai des Tuileries SE-NO',
    'Pont des Invalides (couloir bus)': 'Pont des Invalides',
    '69 Boulevard Ornano (temporaire)': '69 Boulevard Ornano',
    '30 rue Saint Jacques (temporaire)': '30 rue Saint Jacques',
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos NO-SE': '27 quai de la Tournelle NO-SE',
    '27 quai de la Tournelle 27 quai de la Tournelle Vélos SE-NO': '27 quai de la Tournelle SE-NO',
    'Pont des Invalides (couloir bus) N-S': 'Pont des Invalides N-S',
})

# Drop des 2 observations des compteurs dont la direction n'est pas claire
df_clean = df_clean.drop(index=df.loc[df['Nom du compteur'] == '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike OUT]'].index)
df_clean = df_clean.drop(index=df.loc[df['Nom du compteur'] == '10 avenue de la Grande Armée 10 avenue de la Grande Armée [Bike IN]'].index)

df_clean = df_clean.drop(columns=["Nom du compteur"])

# Convertir la colonne en datetime (avec gestion du fuseau horaire)
df_clean["Date et heure de comptage"] = pd.to_datetime(df_clean["Date et heure de comptage"], utc=True)
df_clean["Date et heure de comptage"] = df_clean["Date et heure de comptage"].dt.tz_convert("Europe/Paris")

# Mets les dates dans l'ordre croissant
df_clean = df_clean.sort_values(by="Date et heure de comptage")

#Crée 4 colonnes Année, Jour, Mois et Heure
#Ensuite on range dans l'ordre Heure Jour Mois et Année
df_clean["Jour"] = df_clean["Date et heure de comptage"].dt.day
df_clean["Année"] = df_clean["Date et heure de comptage"].dt.year
df_clean["Mois"] = df_clean["Date et heure de comptage"].dt.month
df_clean["Heure"] = df_clean["Date et heure de comptage"].dt.hour
df_clean["Jour_semaine"] = df_clean["Date et heure de comptage"].dt.weekday + 1
df_clean["Week-end"] = (df_clean["Jour_semaine"] >= 5).astype(int)
df_clean.insert(3, "Jour", df_clean.pop("Jour"))
df_clean.insert(4, "Mois", df_clean.pop("Mois"))
df_clean.insert(5, "Année", df_clean.pop("Année"))
df_clean.insert(6, "Heure", df_clean.pop("Heure"))
#
# Création manuelle des jours fériés en France pour 2023, 2024 et 2025
jours_feries_france = [
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18", "2023-05-29",
    "2023-07-14", "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-09", "2024-05-20",
    "2024-07-14", "2024-08-15", "2024-11-01", "2024-11-11", "2024-12-25",
    "2025-01-01"
]

jours_feries_france = pd.to_datetime(jours_feries_france)

# Ajout de la colonne "Jour férié" (1 si jour férié, 0 sinon)
df_clean["Jour férié"] = df_clean["Date et heure de comptage"].dt.date.isin(jours_feries_france.date).astype(int)

# Définition des périodes de vacances scolaires (Zone C)
vacances = [
    ("2022-12-17", "2023-01-03"),
    ("2023-02-18", "2023-03-06"),
    ("2023-04-22", "2023-05-09"),
    ("2023-07-08", "2023-09-04"),
    ("2023-10-21", "2023-11-06"),
    ("2023-12-24", "2024-01-08"),
    ("2024-02-10", "2024-02-26"),
    ("2024-04-06", "2024-04-22"),
    ("2024-05-08", "2024-05-13"),
    ("2024-07-06", "2024-09-02"),
    ("2024-10-19", "2024-11-04"),
    ("2024-12-21", "2025-01-06"),
]

# Création d'un ensemble de dates correspondant aux vacances
dates_vacances = pd.DatetimeIndex([])  # Initialisation

for debut, fin in vacances:
    dates_vacances = dates_vacances.union(pd.date_range(start=debut, end=fin))

# Appliquer le fuseau horaire pour correspondre à df_clean
dates_vacances = dates_vacances.tz_localize("Europe/Paris")  # On ajoute le TZ

# Vérification d'appartenance
df_clean["Vacances scolaires"] = df_clean["Date et heure de comptage"].dt.normalize().isin(dates_vacances).astype(int)

# Comme on fusionne plusieurs compteurs en un seul lieu,
# on somme le comptage horaire pour chaque compteur d'un même lieu
df_clean = df_clean.groupby(by=['Nom du site de comptage', 'Date et heure de comptage', 'Jour', 'Mois','Année', 'Heure', 'Jour_semaine', 'Week-end', 'Jour férié','Vacances scolaires'], as_index=False)['Comptage horaire'].sum()

assert df_clean.duplicated().sum() == 0

# Nettoyage de valeurs aberrantes
df_clean = df_clean.sort_values(by="Comptage horaire", ascending=False).iloc[5:]

# Suite à observation que la colonne week-end est très corrélée avec la colonne jour_semaine, on drop la colonne week-end
df_clean = df_clean.drop(columns = ["Date et heure de comptage", "Week-end"])

# On split les features et la variable cible
X = df_clean.drop(columns=["Comptage horaire"])
y = df_clean["Comptage horaire"]

# OneHotEncoding des features pour la modélisation
cat_columns = ["Nom du site de comptage", "Jour", "Mois", "Année", "Heure", "Jour_semaine", "Jour férié", "Vacances scolaires"]
encoder = OneHotEncoder(sparse_output=False, dtype=int)
array = encoder.fit_transform(X)
encoded_X = pd.DataFrame(array, columns=encoder.get_feature_names_out(cat_columns))
encoded_X.index = X.index

# Sauvegarde du dataset
encoded_df = pd.concat([X, y], axis=1)
encoded_df.to_csv("data/processed/lieu-compteur-one-hot-encoded.csv")

# Sauvegarde du onehotencoder
joblib.dump(encoder, "models/one_hot_encoder.pkl")
