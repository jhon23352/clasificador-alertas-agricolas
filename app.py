import streamlit as st
import joblib

# Cargar modelo y vectorizador
model = joblib.load("modelo_alertas.pkl")
vectorizer = joblib.load("vectorizador.pkl")

st.title("🌱 Clasificador de Alertas Agrícolas")
st.write("Este prototipo identifica si una alerta pertenece a: helada, sequía, plaga o inundación.")

texto = st.text_area("✍️ Ingresar alerta agrícola:")

if st.button("Clasificar"):
    vector = vectorizer.transform([texto])
    prediccion = model.predict(vector)[0]
    st.success(f"✅ La alerta corresponde a: **{prediccion.upper()}**")
