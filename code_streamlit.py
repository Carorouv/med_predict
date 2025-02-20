import streamlit as st
from PIL import Image
import requests
import joblib
from io import BytesIO
import numpy as np

import warnings 
warnings.filterwarnings("ignore")

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="BioTechInsight", page_icon=":microscope:")

# Mise en place du background de l'application
# image_url = "https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/wallpaper_3.png"

# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url("{image_url}") no-repeat center center fixed;
#         background-size: cover;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Logo centré en haut à gauche
st.sidebar.image(Image.open('Logo_BioTechInsights_détouré.png'), width=280, use_column_width=False)

# Maladies cardiaques 

# Chargement du modèle sauvegardé
model_heart_diseases_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/Gaussian_HeartDisease.joblib'

response_model = requests.get(model_heart_diseases_url)
response_model.raise_for_status()
model_heart_diseases = joblib.load(BytesIO(response_model.content))

# Chargement du scaler sauvegardé
scaler_heart_diseases_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/ScalePwrTransf_HeartDisease.joblib'

response_scaler = requests.get(scaler_heart_diseases_url)
response_scaler.raise_for_status()
scaler_heart_diseases = joblib.load(BytesIO(response_scaler.content))

def maladies_cardiaques():
    st.write('### Maladies cardiaques : renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Champs pour la saisie des données avec une police plus grande
    st.markdown('<div style="font-size: 24px"><label for="age">Âge:</label></div>', unsafe_allow_html=True)
    age = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="age")
    st.markdown('<div style="font-size: 24px"><label for="sex">Sexe:</label></div>', unsafe_allow_html=True)
    sex = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="sex") 
    st.markdown('<div style="font-size: 24px"><label for="chest_pain">Douleur thoracique:</label></div>', unsafe_allow_html=True)
    chest_pain = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="chest_pain") 
    st.markdown('<div style="font-size: 24px"><label for="trestbps">Pression artérielle au repos:</label></div>', unsafe_allow_html=True)
    trestbps = st.number_input("", value=0, min_value=0, max_value=999, step=1, format="%03d", key="trestbps") 
    st.markdown('<div style="font-size: 24px"><label for="chol">Cholestérol sérique en mg/dl:</label></div>', unsafe_allow_html=True)
    chol = st.number_input("", value=0, min_value=0, max_value=999, step=1, format="%03d", key="chol")
    st.markdown('<div style="font-size: 24px"><label for="fbs">Taux de sucre dans le sang à jeun > 120 mg/dl:</label></div>', unsafe_allow_html=True)
    fbs = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="fbs") 
    st.markdown('<div style="font-size: 24px"><label for="restecg">Résultats électrocardiographiques au repos:</label></div>', unsafe_allow_html=True)
    restecg = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="restecg")
    st.markdown('<div style="font-size: 24px"><label for="thalach">Fréquence cardiaque maximale atteinte:</label></div>', unsafe_allow_html=True)
    thalach = st.number_input("", value=0, min_value=0, max_value=999, step=1, format="%03d", key="thalach")
    st.markdown("<div style='font-size: 24px'><label for='exang'>Angine induite par l'exercice:</label></div>", unsafe_allow_html=True)
    exang = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="exang")
    st.markdown("<div style='font-size: 24px'><label for='oldpeak'>Dépression du segment ST induite par l'exercice par rapport au repos:</label></div>", unsafe_allow_html=True)
    oldpeak = st.number_input("", value=0.0, step=0.1, format="%.1f", key="oldpeak")
    st.markdown("<div style='font-size: 24px'><label for='slope'>Pente du segment ST à l'exercice:</label></div>", unsafe_allow_html=True)
    slope = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="slope")
    st.markdown('<div style="font-size: 24px"><label for="ca">Nombre de vaisseaux principaux colorés par la fluoroscopie:</label></div>', unsafe_allow_html=True)
    ca = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="ca")
    st.markdown('<div style="font-size: 24px"><label for="thal">Résultat thallium scintigraphique:</label></div>', unsafe_allow_html=True)
    thal = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="thal")

    # Bouton pour Diagnostic des données
    if st.button("Diagnostic"):
        # Vérification si toutes les entrées sont nulles
        if all(value == 0 for value in [age, sex, chest_pain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
            st.warning("Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
        else:
            # Création du tableau NumPy avec les données saisies par l'utilisateur
            my_data = np.array([[age, sex, chest_pain, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            # Normalisation des données avec le scaler PowerTransformer
            my_data_scaled = scaler_heart_diseases.transform(my_data)
            # Prédictions avec le modèle de cancer du sein CatBoost entraîné
            predictions = model_heart_diseases.predict(my_data_scaled)
            # Affichage du résultat de la prédiction
            
            # Affichage du résultat de la prédiction
            if predictions[0] == 1:
                st.write("Le modèle prédit que votre patient ne présente pas de risque de développer une maladie cardiaque.")
            else:
                st.write("Le modèle prédit que votre patient présente un risque de développer une maladie cardiaque.")

# Maladies du foie

# Chargement du modèle sauvegardé
model_liver_diseases_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/RandomForest_Liver.joblib'

response_model = requests.get(model_liver_diseases_url)
response_model.raise_for_status()
model_liver_diseases = joblib.load(BytesIO(response_model.content))

# Chargement du scaler sauvegardé
scaler_liver_diseases_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/MaxAbsScaler_Liver.joblib'

response_scaler = requests.get(scaler_liver_diseases_url)
response_scaler.raise_for_status()
scaler_liver_diseases = joblib.load(BytesIO(response_scaler.content))

def maladies_du_foie():
    st.write('### Maladies du foie : renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Champs pour la saisie des données avec une police plus grande
    st.markdown('<div style="font-size: 24px"><label for="age">Âge:</label></div>', unsafe_allow_html=True)
    age = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="age")
    st.markdown('<div style="font-size: 24px"><label for="sex">Sexe:</label></div>', unsafe_allow_html=True)
    sex = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="sex") 
    st.markdown('<div style="font-size: 24px"><label for="total_bilirubin">Bilirubine totale:</label></div>', unsafe_allow_html=True)
    total_bilirubin = st.number_input("", value=0.0, min_value=0.0, max_value=99.9, step=0.1, format="%.1f", key="total_bilirubin") 
    st.markdown('<div style="font-size: 24px"><label for="alkaline_phosphotase">Phosphatase alcaline:</label></div>', unsafe_allow_html=True)
    alkaline_phosphotase = st.number_input("", value=0, min_value=0, max_value=999, step=1, format="%d", key="alkaline_phosphotase") 
    st.markdown('<div style="font-size: 24px"><label for="alamine_aminotransferase">Alamine aminotransférase:</label></div>', unsafe_allow_html=True)
    alamine_aminotransferase = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%d", key="alamine_aminotransferase")
    st.markdown('<div style="font-size: 24px"><label for="albumin_and_globulin_ratio">Rapport albumine et globuline:</label></div>', unsafe_allow_html=True)
    albumin_and_globulin_ratio = st.number_input("", value=0.0, min_value=0.0, max_value=9.99, step=0.01, format="%.2f", key="albumin_and_globulin_ratio") 

    # Bouton pour Diagnostic des données
    if st.button("Diagnostic"):
        # Vérification si toutes les entrées sont nulles
        if all(value == 0 for value in [age, sex, total_bilirubin, alkaline_phosphotase, alamine_aminotransferase, albumin_and_globulin_ratio]):
            st.warning("Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
        else:
            # Création du tableau NumPy avec les données saisies par l'utilisateur
            my_data = np.array([[age, sex, total_bilirubin, alkaline_phosphotase, alamine_aminotransferase, albumin_and_globulin_ratio]])
            # Normalisation des données avec le scaler PowerTransformer
            my_data_scaled = scaler_liver_diseases.transform(my_data)
            # Prédictions avec le modèle de cancer du sein CatBoost entraîné
            predictions = model_liver_diseases.predict(my_data_scaled)
            # Affichage du résultat de la prédiction
            if predictions[0] == 0:
                st.write("Le modèle prédit que votre patient ne présente pas de risque de développer une maladie du foie.")
            else:
                st.write("Le modèle prédit que votre patient présente un risque de développer une maladie du foie.")

# Maladie rénale chronique

# Chargement du modèle sauvegardé
model_ckd_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/RandomForest_CKD.joblib'

response = requests.get(model_ckd_url)
response.raise_for_status()
model_ckd = joblib.load(BytesIO(response.content))

# Chargement du scaler sauvegardé
scaler_ckd_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/StandardScaler_CKD.joblib'

response = requests.get(scaler_ckd_url)
response.raise_for_status()
scaler_ckd = joblib.load(BytesIO(response.content))

# Définition d'une fonction pour la page "Cancer du sein"
def maladie_renale_chronique():
    st.write('### Maladie rénale chronique : renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Champs pour la saisie des données avec une police plus grande
    st.markdown('<div style="font-size: 24px"><label for="age">Âge:</label></div>', unsafe_allow_html=True)
    age = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="age")
    st.markdown('<div style="font-size: 24px"><label for="specific_gravity">Gravité spécifique:</label></div>', unsafe_allow_html=True)
    specific_gravity = st.number_input("", value=0.000, step=0.0001, format="%.4f", key="specific_gravity")
    st.markdown('<div style="font-size: 24px"><label for="albumin">Albumine:</label></div>', unsafe_allow_html=True)
    albumin = st.number_input("", value=0.0, step=0.1, format="%.1f", key="albumin")
    st.markdown('<div style="font-size: 24px"><label for="sugar">Sucre:</label></div>', unsafe_allow_html=True)
    sugar = st.number_input("", value=0.0, step=0.1, format="%.1f", key="sugar")
    st.markdown('<div style="font-size: 24px"><label for="red_blood_cells">Globules rouges:</label></div>', unsafe_allow_html=True)
    red_blood_cells = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="red_blood_cells")
    st.markdown('<div style="font-size: 24px"><label for="pus_cells">Cellules de pus:</label></div>', unsafe_allow_html=True)
    pus_cells = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="pus_cells")
    st.markdown('<div style="font-size: 24px"><label for="pus_cells_clumps">Amas de cellules de pus:</label></div>', unsafe_allow_html=True)
    pus_cells_clumps = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="pus_cells_clumps")
    st.markdown('<div style="font-size: 24px"><label for="bacterias">Bactéries:</label></div>', unsafe_allow_html=True)
    bacterias = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="bacterias")
    st.markdown('<div style="font-size: 24px"><label for="blood_glucose_random">Glycémie aléatoire:</label></div>', unsafe_allow_html=True)
    blood_glucose_random = st.number_input("", value=0.0, min_value=0.0, max_value=999.9, step=0.1, format="%.1f", key="blood_glucose_random")
    st.markdown('<div style="font-size: 24px"><label for="blood_urea">Urée sanguine:</label></div>', unsafe_allow_html=True)
    blood_urea = st.number_input("", value=0.0, min_value=0.0, max_value=99.9, step=0.1, format="%.1f", key="blood_urea")
    st.markdown('<div style="font-size: 24px"><label for="serum_creatinine">Créatinine sérique:</label></div>', unsafe_allow_html=True)
    serum_creatinine = st.number_input("", value=0.0, min_value=0.0, max_value=9.9, step=0.1, format="%.1f", key="serum_creatinine")
    st.markdown('<div style="font-size: 24px"><label for="sodium">Sodium:</label></div>', unsafe_allow_html=True)
    sodium = st.number_input("", value=0.0, min_value=0.0, max_value=999.9, step=0.1, format="%.1f", key="sodium")
    st.markdown('<div style="font-size: 24px"><label for="potassium">Potassium:</label></div>', unsafe_allow_html=True)
    potassium = st.number_input("", value=0.0, min_value=0.0, max_value=9.9, step=0.1, format="%.1f", key="potassium")
    st.markdown('<div style="font-size: 24px"><label for="haemoglobin">Hémoglobine:</label></div>', unsafe_allow_html=True)
    haemoglobin = st.number_input("", value=0.0, min_value=0.0, max_value=99.9, step=0.1, format="%.1f", key="haemoglobin")
    st.markdown('<div style="font-size: 24px"><label for="white_blood_cells_count">Numération des globules blancs:</label></div>', unsafe_allow_html=True)
    white_blood_cells_count = st.number_input("", value=0.0, min_value=0.0, max_value=9.9, step=0.1, format="%.1f", key="white_blood_cells_count")
    st.markdown('<div style="font-size: 24px"><label for="red_blood_cells_count">Numération des globules rouges:</label></div>', unsafe_allow_html=True)
    red_blood_cells_count = st.number_input("", value=0.0, min_value=0.0, max_value=9.9, step=0.1, format="%.1f", key="red_blood_cells_count")
    st.markdown('<div style="font-size: 24px"><label for="hypertension">Hypertension:</label></div>', unsafe_allow_html=True)
    hypertension = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="hypertension") 
    st.markdown('<div style="font-size: 24px"><label for="diabetes_mellitus">Diabète sucré:</label></div>', unsafe_allow_html=True)
    diabetes_mellitus = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="diabetes_mellitus")
    st.markdown('<div style="font-size: 24px"><label for="coronary_artery_disease">Maladie coronarienne:</label></div>', unsafe_allow_html=True)
    coronary_artery_disease = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="coronary_artery_disease")
    st.markdown('<div style="font-size: 24px"><label for="appetite">Appétit:</label></div>', unsafe_allow_html=True)
    appetite = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="appetite")
    st.markdown('<div style="font-size: 24px"><label for="pedal_edema">Œdème des membres inférieurs:</label></div>', unsafe_allow_html=True)
    pedal_edema = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="pedal_edema")
    st.markdown('<div style="font-size: 24px"><label for="anemia">Anémie:</label></div>', unsafe_allow_html=True)
    anemia = st.number_input("", value=0, min_value=0, max_value=9, step=1, format="%d", key="anemia")

    # Bouton pour Diagnostic des données
    if st.button("Diagnostic"):
        # Vérification si toutes les entrées sont nulles
        if all(value == 0 for value in [age, specific_gravity, albumin, sugar, red_blood_cells, pus_cells, pus_cells_clumps, bacterias, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin, white_blood_cells_count, red_blood_cells_count, hypertension, diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia]):
            st.warning("Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
        else:
            # Création du tableau NumPy avec les données saisies par l'utilisateur
            my_data = np.array([[age, specific_gravity, albumin, sugar, red_blood_cells, pus_cells, pus_cells_clumps, bacterias, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin, white_blood_cells_count, red_blood_cells_count, hypertension, diabetes_mellitus, coronary_artery_disease, appetite, pedal_edema, anemia]])
            # Normalisation des données avec le scaler PowerTransformer
            my_data_scaled = scaler_ckd.transform(my_data)
            # Prédictions avec le modèle de cancer du sein CatBoost entraîné
            predictions = model_ckd.predict(my_data_scaled)
            # Affichage du résultat de la prédiction
            
            # Affichage du résultat de la prédiction
            if predictions[0] == 0:
                st.write("Le modèle prédit que le patient présente un risque de développer la maladie rénale chronique.")
            else:
                st.write("Le modèle prédit que le patient ne présente pas de risque de développer la maladie rénale chronique.")

# Diabète

# Chargement du modèle sauvegardé
model_diabetes_diseases_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/RandomForest_Diabetes.joblib'

response_model = requests.get(model_diabetes_diseases_url)
response_model.raise_for_status()
model_diabetes = joblib.load(BytesIO(response_model.content))

def diabete():
    st.write('### Diabète : renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Champs pour la saisie des données avec une police plus grande
    st.markdown('<div style="font-size: 24px"><label for="pregnancies">Nombre de grossesses:</label></div>', unsafe_allow_html=True)
    pregnancies = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="pregnancies")
    st.markdown('<div style="font-size: 24px"><label for="glucose">Niveau de glucose:</label></div>', unsafe_allow_html=True)
    glucose = st.number_input("", value=0, min_value=0, max_value=999, step=1, format="%03d", key="glucose") 
    st.markdown('<div style="font-size: 24px"><label for="blood_pressure">Pression artérielle:</label></div>', unsafe_allow_html=True)
    blood_pressure = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="blood_pressure") 
    st.markdown('<div style="font-size: 24px"><label for="skin_thickness">Épaisseur de la peau:</label></div>', unsafe_allow_html=True)
    skin_thickness = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="skin_thickness") 
    st.markdown('<div style="font-size: 24px"><label for="BMI">Indice de masse corporelle (BMI):</label></div>', unsafe_allow_html=True)
    BMI = st.number_input("", value=0.0, step=0.1, format="%.1f", key="BMI")
    st.markdown('<div style="font-size: 24px"><label for="diabetes_pedigree_function">Fonction de pédigrée diabétique:</label></div>', unsafe_allow_html=True)
    diabetes_pedigree_function = st.number_input("", value=0.000, step=0.001, format="%.3f", key="diabetes_pedigree_function") 
    st.markdown('<div style="font-size: 24px"><label for="age">Âge:</label></div>', unsafe_allow_html=True)
    age = st.number_input("", value=0, min_value=0, max_value=99, step=1, format="%02d", key="age")

    # Bouton pour Diagnostic des données
    if st.button("Diagnostic"):
        # Vérification si toutes les entrées sont nulles
        if all(value == 0 for value in [pregnancies, glucose, blood_pressure, skin_thickness, BMI, diabetes_pedigree_function, age]):
            st.warning("Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
        else:
            # Création du tableau NumPy avec les données saisies par l'utilisateur
            my_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, BMI, diabetes_pedigree_function, age]])
            # Prédictions avec le modèle de cancer du sein CatBoost entraîné
            predictions = model_diabetes.predict(my_data)
            # Affichage du résultat de la prédiction
            if predictions[0] == 0:
                st.write("Le modèle prédit que votre patient ne présente pas de risque de développer du diabète.")
            else:
                st.write("Le modèle prédit que votre patient présente un risque de développer du diabète.")

# Cancer du sein

# Chargement du modèle sauvegardé
model_cancer_breast_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/CastBoost_Cancerbreast.joblib'

response = requests.get(model_cancer_breast_url)
response.raise_for_status()
model_cancer_breast = joblib.load(BytesIO(response.content))

# Chargement du scaler sauvegardé
ScalePwrTransf_Cancerbreast_url = 'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/ScalePwrTransf_Cancerbreast.joblib'

response = requests.get(ScalePwrTransf_Cancerbreast_url)
response.raise_for_status()
scaler_cancer_breast = joblib.load(BytesIO(response.content))

# Définition d'une fonction pour la page "Cancer du sein"
def cancer_du_sein():
    st.write('### Cancer du sein : renseignez les biomarqueurs de votre patient(e) et lancez le traitement')

    # Champs pour la saisie des données avec une police plus grande
    st.markdown('<div style="font-size: 24px"><label for="mean_radius">Rayon moyen de la cellule:</label></div>', unsafe_allow_html=True)
    mean_radius = st.number_input("", value=0.00, step=0.01, format="%.2f", key="mean_radius")    
    st.markdown('<div style="font-size: 24px"><label for="mean_texture">Texture moyenne de la cellule:</label></div>', unsafe_allow_html=True)
    mean_texture = st.number_input("", value=0.00, step=0.01, format="%.2f", key="mean_texture") 
    st.markdown('<div style="font-size: 24px"><label for="mean_smoothness">Régularité moyenne de la cellule:</label></div>', unsafe_allow_html=True)
    mean_smoothness = st.number_input("", value=0.00000, step=0.00001, format="%.5f", key="mean_smoothness") 
    st.markdown('<div style="font-size: 24px"><label for="mean_compactness">Compacité moyenne de la cellule:</label></div>', unsafe_allow_html=True)
    mean_compactness = st.number_input("", value=0.00000, step=0.00001, format="%.5f", key="mean_compactness") 
    st.markdown('<div style="font-size: 24px"><label for="mean_concavity">Concavité moyenne de la cellule:</label></div>', unsafe_allow_html=True)
    mean_concavity = st.number_input("", value=0.00000, step=0.00001, format="%.5f", key="mean_concavity")
    st.markdown('<div style="font-size: 24px"><label for="mean_concave_points">Moyenne des points de concavité de la cellule:</label></div>', unsafe_allow_html=True)
    mean_concave_points = st.number_input("", value=0.00000, step=0.00001, format="%.5f", key="mean_concave_points") 
    st.markdown('<div style="font-size: 24px"><label for="mean_symmetry">Symétrie moyenne de la cellule:</label></div>', unsafe_allow_html=True)
    mean_symmetry = st.number_input("", value=0.0000, step=0.0001, format="%.4f", key="mean_symmetry")
    st.markdown('<div style="font-size: 24px"><label for="mean_fractal_dimension">Dimension fractale moyenne de la cellule:</label></div>', unsafe_allow_html=True)
    mean_fractal_dimension = st.number_input("", value=0.00000, step=0.00001, format="%.5f", key="mean_fractal_dimension")


    # Bouton pour Diagnostic des données
    if st.button("Diagnostic"):
        # Vérification si toutes les entrées sont nulles
        if all(value == 0 for value in [mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]):
            st.warning("Veuillez renseigner les biomarqueurs de vos patients pour pouvoir faire une prédiction.")
        else:
            # Création du tableau NumPy avec les données saisies par l'utilisateur
            my_data = np.array([[mean_radius, mean_texture, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]])
            # Normalisation des données avec le scaler PowerTransformer
            my_data_scaled = scaler_cancer_breast.transform(my_data)
            # Prédictions avec le modèle de cancer du sein CatBoost entraîné
            predictions = model_cancer_breast.predict(my_data_scaled)
            # Affichage du résultat de la prédiction
            
            # Affichage du résultat de la prédiction
            if predictions[0] == 1:
                st.write("Le modèle prédit que les cellules analysées ne sont pas cancéreuses.")
            else:
                st.write("Le modèle prédit que les cellules analysées sont cancéreuses.")

# Ajoutez le code pour afficher les boutons de navigation
st.sidebar.markdown('<style>.sidebar .sidebar-content { width: 100%; } .sidebar .sidebar-content .block-container {display: none;}</style>', unsafe_allow_html=True)
selected_page = st.sidebar.radio("", ["Accueil", "Maladies cardiaques", "Maladies du foie", "Maladie rénale chronique", "Diabète", "Cancer du sein"])

# Contenu des boutons
if selected_page == "Accueil":
    # Titre et premier paragraphe
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 36px;"><strong>Bienvenue dans votre application de prédiction et de prévention des risques liés à la santé</strong></h1>
        <p style="font-size: 20px;">Cette application vous offre la possibilité de renseigner les biomarqueurs de vos patients afin d'évaluer leur risque 
        de développer diverses pathologies dont le cancer du sein, les maladies cardio-vasculaires, le diabète, les affections 
        hépatiques ou la maladie rénale chronique :</p>
    </div>
    """, unsafe_allow_html=True)

    # Affichage des icônes côte à côte avec une taille ajustée
    icons = [
        'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/coeur.png',

        'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/diabete.png',

        'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/foie.png',

        'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/reins.png',

        'https://raw.githubusercontent.com/Carorouv/bio-tech-insights/main/sein.png'

    ]

    # Taille ajustée des icônes
    icon_size = 100

    # Affichage des icônes centrées avec une légende
    icons_html = ''.join(
        f'<div style="display: inline-block; text-align: center; margin: 0 20px;">'
        f'   <img src="{icon}" width="{icon_size}" alt="{caption}">'
        f'   <p style="margin-top: 10px;">{caption}</p>'
        f'</div>'
        for icon, caption in zip(icons, ['Cœur', 'Diabète', 'Foie', 'Reins', 'Sein'])
    )

    # Utilisation de st.markdown pour afficher les icônes centrées
    st.markdown(icons_html, unsafe_allow_html=True)

    # Dernier paragraphe
    st.markdown("""
    <div style="text-align: center;">
    <p style="font-size: 20px;">Ces prédictions ne sauraient en aucun cas remplacer votre avis de professionnel de la santé et n'ont vocation qu'à vous fournir un 
    soutien supplémentaire dans le processus de prise de décision concernant d'éventuels examens complémentaires et traitements.</p>
    <p style="font-size: 20px;">Les données que vous saisissez sont entièrement anonymisées et ne sont pas conservées, en stricte conformité avec le Règlement Général de 
    Protection des Données (RGPD).</p>
    </div>
    """, unsafe_allow_html=True)
    
elif selected_page == "Maladies cardiaques":
    maladies_cardiaques()
elif selected_page == "Maladies du foie":
    maladies_du_foie()
elif selected_page == "Maladie rénale chronique":
    maladie_renale_chronique()
elif selected_page == "Diabète":
    diabete()
elif selected_page == "Cancer du sein":
    cancer_du_sein()