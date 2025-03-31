import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv("lieu-compteur-one-hot-encoded.csv")

#Image
st.image("banniere6.jpeg", use_column_width=True)

#Titres
st.markdown("""
# **Data analyse du trafic cycliste à Paris**  
### _de Janvier 2023 à Février 2025_
""")
st.sidebar.title("Sommaire")
pages=["Présentation du Projet", "Dataset", "Pre-processing", "Visualisation des données", "Modèles de classification", "Modèles de régression", "Interprétation et résultats", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

import streamlit as st


if page == pages[0]:
    st.markdown("<h2 style='font-size: 36px;'>I. Contexte</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 20px;'>
    Face à l'essor du vélo comme mode de transport durable, la Ville de Paris a mis en place, depuis plusieurs années, un réseau de compteurs à vélo permanents pour mesurer l'évolution de la pratique cycliste dans la capitale.
    <br><br>
    Ces capteurs, installés sur les axes clés de la ville, collectent en continu des données sur le flux des cyclistes.
    </div>
    """, unsafe_allow_html=True)

    # Ajout d'espace avant l'image
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.image("comptagevélo.jpeg", use_column_width=True)

    # Ajout d'espace après l'image
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 20px;'>
    Ce projet s'inscrit dans une démarche de transition vers une mobilité plus verte et une volonté d'adapter les infrastructures urbaines aux besoins réels, tel que proposé dans le plan vélo 2021-2026 d'aménagement de pistes cyclables de la mairie de Paris.
    <br><br>
    L'enjeu est de transformer ces données brutes en insights exploitables, permettant d'éclairer les décisions publiques de manière objective et data-driven.
    </div>
    """, unsafe_allow_html=True)




st.markdown("---")


if page == pages[0]: 

    st.markdown("<h2 style='font-size: 36px; margin-bottom: 24px;'>II. Objectifs</h2>", unsafe_allow_html=True)


    st.markdown("""
<div style='font-size: 30px; font-family: Inter, sans-serif;'>

<p>Ce projet vise à développer un <strong>outil prédictif du trafic cycliste à Paris</strong>, en exploitant les données historiques des compteurs vélo.</p>

<div style='margin-top: 30px;'>
<strong>Objectifs principaux :</strong>
<ul>
  <li>Identifier les <strong>tendances d’usage</strong> (heures de pointe, zones saturées, variations saisonnières).</li>
  <li>Générer des <strong>visualisations claires</strong> (cartes thermiques, graphiques temporels).</li>
  <li>Aider à la <strong>prise de décision</strong> sur les aménagements à prioriser.</li>
</ul>
</div>

<div style='margin-top: 30px;'>
<strong>Bénéfices pour la Mairie de Paris :</strong>
<ul>
  <li>Prioriser les <strong>aménagements ciblés</strong> (pistes élargies, carrefours sécurisés, nouveaux itinéraires).</li>
  <li>Évaluer l’<strong>impact des politiques existantes</strong>.</li>
  <li><strong>Optimiser le réseau cyclable</strong> à long terme.</li>
</ul>
</div>

<div style='margin-top: 30px;'>
<strong>Ambition finale :</strong>
<blockquote style='border-left: 5px solid #000; padding-left: 10px; margin-top: 10px;'>
Réduire les <strong>conflits d’usage</strong>, améliorer la <strong>sécurité</strong>, et encourager la pratique du vélo grâce à une <strong>planification data-driven</strong>, combinant <strong>rétrospective</strong> et <strong>prédiction</strong> pour une mobilité plus fluide et résiliente.
</blockquote>
</div>

</div>
""", unsafe_allow_html=True)









if page == pages[1]: 

    st.markdown("""
### 1. Source des données""")  
                
    st.image("opendata2.png", use_column_width=True)

    st.markdown("""
- Utilisation des jeux de données ouverts proposés par la Ville de Paris via opendata.paris.fr :
                 
- Données publiées sous la licence Open Database License (ODbL), qui autorise la réutilisation, l’adaptation et la création de travaux dérivés à partir de ces jeux de données, à condition d’en citer la source.

---
                
### 2. Période 
                
Les données sont mises à jour quotidiennement. 

Nous avons récupéré toutes les données du 1er janvier 2023 au 28 février 2025 (26 mois).                

---

### 3. Contenu des jeux de données  
Les jeux de données recensent les comptages horaires de vélos effectués par environ une centaine de compteurs répartis dans Paris.
                  
Chaque lieu de comptage est généralement équipé de deux compteurs, positionnés pour mesurer le trafic dans chaque direction d’une même rue. 
                 
Au total, pour la période 2023-2024, le jeu de données contient environ 1,8 million de lignes réparties sur 16 variables.

---

### 4. Structure des données  
Chaque ligne du dataset correspond au nombre de vélos enregistrés pendant une heure par un compteur donné.  
Les données incluent, en plus du comptage horaire, plusieurs métadonnées associées au compteur ou au site de comptage, telles que :
- Le nom et l’identifiant du compteur  
- Le lieu de comptage  
- La date d’installation  
- Les coordonnées géographiques  
- Éventuellement, des liens vers des photos du site

---

### 5. Nettoyage et sélection des variables  
Afin de simplifier et d’optimiser l’analyse, nous avons supprimé les variables non pertinentes pour l'entraînement du modèle, en particulier les champs techniques ou visuels comme les liens vers les photos, les identifiants internes ou le type d’image.  

Voici un extrait de notre dataset avec les variables que nous avons décidé de conserver :""")

    st.image("dataframe.jpeg", use_column_width=True)

    st.markdown("""
---

### 6. Objectif d’analyse et variable cible  
L’objectif de notre étude est de prédire le nombre de vélos comptés pendant une heure sur un compteur donné.  
La variable cible de notre modèle est donc le comptage horaire, un indicateur clé pour analyser l'évolution de la circulation cycliste dans Paris.

---

### 7. Forces et limites du dataset  
Le dataset se distingue par sa précision horaire et sa couverture géographique dense, ce qui permet d’identifier des tendances temporelles comme les variations quotidiennes ou saisonnières du trafic cycliste.  
Cependant, il ne contient pas d’informations contextuelles telles que :
- La météo  
- La présence d’événements particuliers (manifestations, grèves, festivals)  
- Ou des données sociodémographiques comme la densité de population par zone  

Cette absence limite la profondeur des analyses prédictives que l’on peut mener.
""")

if page == pages[2]: 

    st.markdown("""
### 1. Suppression des NaN 
    
Certaines variables de métadonnées des compteurs ("Identifiant du compteur", "Coordonnées géographiques", ...) ont des valeurs NaN (environ 3.4% sur le dataset)

Plusieurs compteurs du dataset correspondaient en réalité à un même emplacement, ce qui a permis de réduire les NaN en les renommant et fusionnant. 

Les derniers NaN provenaient de deux compteurs atypiques, finalement supprimés pour obtenir un dataset complet et sans valeurs manquantes

    
### 2. Conversion Date au format datetime
                
Variable "Date et heure de comptage" convertie au format datetime de 
Pandas. (fuseau horaire Europe/Paris afin de correctement capturer 
les tendances journalières) 
                
### 3. Ajout de variables
                
La variable "Date et heure de comptage" décomposée en variables "année", 
"mois", "jour", "jour de la semaine" et "heure" afin de faciliter la data visualisation et voir si 
certaines de ces variables étaient corrélés à notre variable cible. 
 
Ajout des variables catégorielles binaires "Week-end", "Jour fériés" et "Vacances scolaires" afin de mesurer si les jours non travaillés ont un impact sur la pratique cyclable.  

### 4. Normalisation des données
                
Normalisation min-max des données, pour réduire l'impact des valeurs extrêmes de "Comptage horaire" sur les prédictions de notre modèle,  la variable "Comptage horaire" ne suivant pas une loi normale.
 
""")
    st.write("### Extrait du Dataframe")
    st.dataframe(df.head(10))






                
if page == pages[3]: 
    st.write("### Visualisation des données")



    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ### I. Cartographie

    Carte de la ville de Paris représentant les positions des différents compteurs du dataset (La taille de chaque point correspond au comptage horaire total).""")

    st.image("carte.png", use_column_width=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    Compteurs sont répartis sur les axes principaux :
            
    - Sud-Ouest / Nord-Est (avenue Denfert-Rochereau et boulevard Sébastopol) 
    - Est-Ouest (avenue de la Grande Armée et avenue des Champs-Élysées). 
    - Les quais de la Seine ainsi que le boulevard périphérique Sud (le long de la voie de tram T3a) sont aussi couverts. 
                
    Boulevard périphérique nord et les 17 et 18e arrondissements n'ont pas de compteurs. 
    
    Compteurs "centraux" ont plus de comptage que ceux en périphérie de Paris : 
    Corrélation entre la localisation du compteur et le comptage horaire ?


    ### II. Évolution temporelle""")

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown(""" 
    ##### a. Saisonnalité du trafic""")
    
    fig = plt.figure()
    sns.barplot(df, x='Mois', y='Comptage horaire', errorbar=None)
    plt.xlabel("Mois")
    plt.xticks(rotation=45)
    plt.title("Comptage horaire moyen en fonction du mois");
    st.pyplot(fig)

    st.markdown("""
    <div style='font-size:18px;'>
    On constate une baisse du comptage en <strong>hiver</strong> (janvier et décembre) et en <strong>été</strong> (au mois d'août).<br>
    Cela est peut-être dû aux <strong>vacances</strong>, à certains <strong>événements</strong> (JO de Paris en août) et à la <strong>météo</strong> (il fait plus froid en hiver, ce qui n'encourage pas la pratique cycliste).
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ##### b. Comportement selon les jours""")
    
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(df, x='Jour_semaine', y='Comptage horaire', errorbar=None)
    plt.xlabel("Jour du mois")
    plt.title("Comptage horaire moyen en fonction du jour de la semaine");
    st.pyplot(fig)

    st.markdown("""
    <div style='font-size:18px;'>
    On constate également plus de <strong>comptages</strong> du <strong>lundi au vendredi</strong>, ce qui correspond aux <strong>trajets domicile-travail</strong>.<br>
    En moyenne, il y a environ <strong>50% de <strong>vélos</strong> en plus</strong> en <strong>semaine</strong> par rapport au <strong>week-end</strong>.
    </div>
    """, unsafe_allow_html=True)


    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    ##### c. Evolution du trafic au fil des heures""")

    st.markdown("""
    <div style='font-size:18px;'>
    À gauche : <strong>jours de la semaine</strong> (lundi à vendredi) — À droite : <strong>week-end</strong> (samedi et dimanche)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # Filtrage
    df_semaine = df[df['Jour_semaine'].isin([1, 2, 3, 4, 5])]
    df_weekend = df[df['Jour_semaine'].isin([6, 7])]

    # Détermination du même axe Y pour les deux graphiques
    y_max = max(df['Comptage horaire'].max(), 1)  # max global pour fixer l'échelle

    # Création des deux graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    # Graphique semaine
    sns.barplot(ax=axes[0], data=df_semaine, x='Heure', y='Comptage horaire', errorbar=None)
    axes[0].set_title("Comptage horaire moyen (lundi à vendredi)")
    axes[0].set_xlabel("Heure de la journée")
    axes[0].set_ylabel("Comptage horaire")
    axes[0].set_ylim(0, 400)

    # Graphique week-end
    sns.barplot(ax=axes[1], data=df_weekend, x='Heure', y='Comptage horaire', errorbar=None)
    axes[1].set_title("Comptage horaire moyen (week-end)")
    axes[1].set_xlabel("Heure de la journée")
    axes[1].set_ylabel("")  # on peut laisser vide pour alléger visuellement
    axes[1].set_ylim(0, 400)

    st.pyplot(fig)

    st.markdown("""
    <div style='font-size:18px;'>
    <strong>Forte augmentation du trafic</strong> aux <strong>heures de pointe</strong> (8h–9h / 18h–19h) en semaine.<br>
    <strong>Volume de passages</strong> relativement <strong>régulier</strong> entre <strong>11h et 20h</strong> le <strong>week-end</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### III. Corrélation entre les variables""")
    
    st.image("matrice.jpeg", use_column_width=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <strong>Matrice de corrélation entre les variables</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px;'>
    <ul>
    <li>Le <strong>comptage horaire</strong> est légèrement corrélé au <strong>nom du compteur</strong> et à <strong>l'heure de la journée</strong>.</li>
    <li>Corrélation forte entre les variables <strong>jour_semaine</strong> et <strong>week-end</strong> (variables potentiellement redondantes).</li>
    <li>Forte corrélation entre <strong>"date et heure de comptage"</strong>, <strong>année</strong> et <strong>mois</strong>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

   
