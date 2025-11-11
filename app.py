import streamlit as st
import tensorflow as tf
from PIL import Image
import joblib

# =========================================
# CARGAR MODELO DE IMAGEN (TFLite)
# =========================================
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return int(output.argmax())


# =========================================
# CARGAR MODELO DE TEXTO
# =========================================
vectorizer = joblib.load("vectorizador.pkl")
modelo_texto = joblib.load("modelo_texto.pkl")


def predict_text(text):
    vector = vectorizer.transform([text])
    prediction = modelo_texto.predict(vector)[0]
    return prediction


# =========================================
# CARGAR LABELS
# =========================================
with open("labels.txt", "r") as file:
    labels = [l.strip() for l in file.readlines()]


# =========================================
# INTERFAZ STREAMLIT
# =========================================
st.title("🌱 Clasificador de Alertas Agrícolas")
st.write("Clasificación mediante texto o imagen (plaga, sequía, inundación o helada).")

menu = st.radio("Selecciona el tipo de entrada:", ["📝 Texto", "📸 Imagen"])

# ---- TEXTO ----
if menu == "📝 Texto":
    texto = st.text_area("✍️ Escribe la alerta agrícola")
    if st.button("Clasificar texto"):
        if texto.strip() != "":
            resultado = predict_text(texto)
            st.success(f"✅ La alerta corresponde a: **{resultado.upper()}**")
        else:
            st.warning("⚠️ Debes escribir una alerta.")

# ---- IMAGEN ----
else:
    archivo = st.file_uploader("📸 Cargar imagen", type=["jpg", "jpeg", "png"])
    if archivo:
        image = Image.open(archivo)
        st.image(imag
