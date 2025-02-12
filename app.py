import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado (LinearRegression) y el scaler usando joblib
modelo = joblib.load('modelo_regresion.bin')
scaler = joblib.load('scaler.bin')

# Título y autor
st.title('Predictor de Compra de Bicicletas')
st.subheader('Autor: Alfredo Díaz')

# Sidebar para ingresar las variables
st.sidebar.header('Ingresa los datos del cliente')

# Slider para Edad e Ingresos
edad = st.sidebar.slider('Edad', 20, 60, 40)
ingresos = st.sidebar.slider('Ingresos', 30000, 150000, 80000)

# Desplegables para Experiencia y Satisfacción
experiencia = st.sidebar.selectbox('Experiencia', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], index=6)
satisfaccion = st.sidebar.selectbox('Satisfacción', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=5)

# Mostrar las variables ingresadas sin normalizar
st.write("### Datos Ingresados:")
st.write(f"Edad: {edad}")
st.write(f"Ingresos: {ingresos}")
st.write(f"Experiencia: {experiencia}")
st.write(f"Satisfacción: {satisfaccion}")

# Preparar los datos para la predicción
datos_entrada = np.array([[edad, ingresos, experiencia, satisfaccion]])

# Normalizar los datos
datos_normalizados = scaler.transform(datos_entrada)

# Realizar la predicción
prediccion = modelo.predict(datos_normalizados)

# Mostrar el valor de compra predicho
st.write("### Valor Predicho de Compra:")
st.markdown(f"#### **${prediccion[0]:,.2f}**")
