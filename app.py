import streamlit as st


# Cargar modelo y vectorizador
import tensorflow as tf

# Cargar modelo TFLite
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
    return output.argmax()

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
