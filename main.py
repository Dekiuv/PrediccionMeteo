import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, f1_score, mean_absolute_error, mean_squared_error
import joblib

# Configurar la página de Streamlit como primera línea
st.set_page_config(
    page_title="Simulador Meteorológico",
    page_icon="🌤️",
    layout="wide"
)

# Cargar el modelo entrenado
@st.cache_data
def cargar_modelo(nombre_archivo):
    with open(nombre_archivo, 'rb') as f:
        model = joblib.load(f)
    return model

# Cargar los datos de entrada desde SQLite
@st.cache_data
def cargar_datos():
    db_path = "CSV/Prediccion.db"  # Cambiar a la ruta correcta
    connection = sqlite3.connect(db_path)
    query_valores = "SELECT * FROM ValoresLimpios"
    df_valores = pd.read_sql_query(query_valores, connection)
    connection.close()
    return df_valores

# Función para preparar los datos (igual que en entreno.py)
def preparar_datos(df, feature_columns):
    target_column = 'weather_id'  # Columna que se va a predecir

    X = df[feature_columns]  # Características
    y = df[target_column]  # Etiquetas

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)  # Escalar los datos de entrenamiento
    X_test_scaled = scaler.transform(X_test)  # Escalar los datos de prueba

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler

# Cargar el modelo entrenado
modelo_guardado = "ModeloEntrenado.pkl"
best_model = cargar_modelo(modelo_guardado)

# Cargar los datos
df_valores = cargar_datos()

# Características fijas para el modelo
features_options = ['precipitation', 'wind', 'humidity', 'visibility']
selected_features = features_options  # Usamos estas características siempre

# Preparar los datos con las características seleccionadas
X_train, X_test, y_train, y_test, scaler = preparar_datos(df_valores, selected_features)

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico - Grupo 3")
st.write("Esta aplicación permite predecir las condiciones meteorológicas basándose en los datos ingresados.")

# Dividir la página en dos columnas
col1, col2 = st.columns(2)

# Columna 1: Formulario de entrada
with col1:
    precipitation = st.number_input('Precipitación (mm)', min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    wind = st.number_input('Viento (km/h)', min_value=0.0, max_value=150.0, value=10.0, step=0.1)
    humidity = st.number_input('Humedad (%)', min_value=0, max_value=100, value=60, step=1)
    visibility = st.number_input('Visibilidad (km)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    # Botón para hacer la predicción
    predict_button = st.button("Predecir Clima")

# Columna 2: Resultado de la predicción
with col2:
    col1, col2, col3 = st.columns(3)
    with col2:
        if predict_button:
            # Realizar la predicción con el modelo cargado
            sample = [[precipitation, wind, humidity, visibility]]
            sample_scaled = scaler.transform(sample)  # Usar el scaler entrenado
            prediction = best_model.predict(sample_scaled)
            weather_map = {1: "Tormenta", 2: "Lluvia", 3: "Nublado", 4: "Niebla", 5: "Soleado"}
            predicted_weather = weather_map.get(prediction[0], "Desconocido")

            # Mostrar la imagen correspondiente según el resultado de la predicción
            image_paths = {
                "Soleado": "Image/Soleado.png", 
                "Tormenta": "Image/Tormenta.png",
                "Lluvia": "Image/Lluvia.png",
                "Nublado": "Image/Cloudy.png",
                "Niebla": "Image/Fog.png"
            }
            imagen_path = image_paths.get(predicted_weather, "Image/Desconocido.png")
            st.image(imagen_path, width=310)

            # Calculo de métricas
            y_pred = best_model.predict(X_test)  # Predicciones en el conjunto de prueba
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Mostrar las métricas en la consola
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-score: {f1:.2f}")
            print(f"R² del modelo: {r2:.2f}")
            print(f"MAE (Error Absoluto Medio): {mae:.2f}")
            print(f"MSE (Error Cuadrático Medio): {mse:.2f}")