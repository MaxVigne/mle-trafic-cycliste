import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Image
st.image("trafic2.jpeg", use_column_width=True)

#Titres
st.markdown("""
# **Data analyse du trafic cycliste à Paris**  
### _de Janvier 2023 à Février 2025_
""")
st.sidebar.title("Sommaire")
pages=["Projet", "Datasets", "Analyse Exploratoire", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)


