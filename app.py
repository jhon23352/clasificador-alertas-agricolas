import streamlit as st


# Cargar modelo y vectorizador
from tensorflow import keras

model = keras.models.load_model("keras_model.h5")

with open("labels.txt", "r") as file:
    labels = file.readlines()

vectorizer = joblib.load("vectorizador.pkl")

st.title("🌱 Clasificador de Alertas Agrícolas")
st.write("Este prototipo identifica si una alerta pertenece a: helada, sequía, plaga o inundación.")

texto = st.text_area("✍️ Ingresar alerta agrícola:")

if st.button("Clasificar"):
    vector = vectorizer.transform([texto])
    prediccion = model.predict(vector)[0]
    st.success(f"✅ La alerta corresponde a: **{prediccion.upper()}**")
