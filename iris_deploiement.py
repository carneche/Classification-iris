import streamlit as st
import pandas as pd
import numpy as np
import joblib

def main():
    st.title("Prédiction de la classe d'un iris")
    st.text("Veuillez entrer les caractéristiques de l'iris dont vous voulez prédire la classe")

    # Lecture des données saisies par l'utilisateur
    sepal_length = st.number_input('Sepal length', min_value=0.0)
    sepal_width = st.number_input('Sepal width', min_value=0.0)
    petal_length = st.number_input('Petal length', min_value=0.0)
    petal_width = st.number_input('Pepal width', min_value=0.0)

    # Chargement du modèle joblib
    knn_from_joblib = joblib.load('knn_model.pkl')  
    
    # Test du modèle pour faire des prédictions 
    value = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    def model_predict(values):
        prediction = knn_from_joblib.predict(values) 
        result = ""
        if prediction == 0: 
            return 'Virginica'
        if prediction == 1:
            return 'Versicolor'
        if prediction == 2:
            return 'Setosa'
    
    prediction = model_predict(value)        
    # Affichag du résultat
    if st.button('Prédire'):
        st.write('Cet iris appartient à la classe :', prediction)

if __name__ == "__main__":
	main()