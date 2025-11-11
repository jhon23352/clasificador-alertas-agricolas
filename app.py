import streamlit as st
import tensorflow as tf
from PIL import Image
import joblib

# ==========================
# CARGA DEL MODELO TFLITE
# ==========================

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    # Preprocesar imagen
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0

    # Inferencia
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return int(output.argmax())

# ==========================
# CARGA DE LABELS
# ==========================
with open("labels.txt", "r") as file:
    labels = [l.strip() for l in file.readlines()]

# ==========================
# INTERFAZ STREAMLIT
# ==========================

st.title("🌱 Clasificador de Alertas Agrícolas por Imagen")

uploaded_file = st.file_uploader("📸 Cargar imagen de la alerta", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Clasificar alerta"):
        pred_idx = predict(image)
        resultado = labels[pred_idx]

        st.success(f"✅ La alerta detectada es: **{resultado.upper()}**")
