import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score
from imblearn.over_sampling import SMOTE
import joblib

# Configurar OMP_NUM_THREADS para optimizar el uso de CPU
os.environ["OMP_NUM_THREADS"] = "4"

# Configuración de Streamlit
st.set_page_config(
    page_title="Simulador Meteorológico",
    page_icon="🌤️",
    layout="wide"
)

# Función para cargar datos desde SQLite
@st.cache_data
def cargar_datos():
    db_path = "CSV/Prediccion.db"  # Cambiar a la ruta correcta
    connection = sqlite3.connect(db_path)
    query_valores = "SELECT * FROM ValoresLimpios"
    df_valores = pd.read_sql_query(query_valores, connection)
    connection.close()
    return df_valores

# Función para preparar los datos
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

# Pesos de clase definidos
class_weights = {
    1: 2.7517886626307098,  # Tormenta
    2: 2.6399155227032733,  # Lluvia
    3: 4.2408821034775235,  # Nublado
    4: 56.81818181818182,   # Niebla
    5: 227.27272727272728   # Soleado
}

# Función para optimizar hiperparámetros y entrenar el modelo
def optimizar_y_entrenar(X_train, y_train):
    param_distributions = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': [class_weights]  # Usar los pesos de clase
    }
    # Optimizar modelo con RandomizedSearchCV
    model = SVC(probability=True)  # Activar las probabilidades
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)

    st.write(f"**Mejores hiperparámetros:** {random_search.best_params_}")
    return random_search.best_estimator_

# Diccionario de mapeo para 'weather_id'
weather_map = {
    1: "Tormenta",
    2: "Lluvia",
    3: "Nublado",
    4: "Niebla",
    5: "Soleado"
}

# Función para guardar el modelo entrenado
def guardar_modelo(modelo, nombre_archivo):
    with open(nombre_archivo, 'wb') as f:
        joblib.dump(modelo, f)
    st.write(f"Modelo guardado en {nombre_archivo}")

# Función para cargar el modelo entrenado
@st.cache_data
def cargar_modelo(nombre_archivo):
    with open(nombre_archivo, 'rb') as f:
        model = joblib.load(f)
    return model

# Cargar datos
df_valores = cargar_datos()

# Características fijas para el modelo
features_options = ['precipitation', 'wind', 'humidity', 'visibility']
selected_features = features_options  # Usamos estas características siempre

# Preparar los datos con las características seleccionadas
X_train, X_test, y_train, y_test, scaler = preparar_datos(df_valores, selected_features)

# Verificar si el modelo ya está guardado
modelo_guardado = "ModeloEntrenado.pkl"
if os.path.exists(modelo_guardado):
    # Cargar el modelo guardado
    best_model = cargar_modelo(modelo_guardado)
else:
    # Entrenar el modelo si no está guardado
    best_model = optimizar_y_entrenar(X_train, y_train)
    # Guardar el modelo entrenado para futuras ejecuciones
    guardar_modelo(best_model, modelo_guardado)
    st.write("Modelo entrenado y guardado exitosamente!")

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico")
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
    if predict_button:
        with st.spinner('Realizando la predicción...'):
            # Realizar la predicción con el modelo cargado o entrenado
            sample = scaler.transform([[precipitation, wind, humidity, visibility]])
            prediction = best_model.predict(sample)
            predicted_weather = weather_map.get(prediction[0], "Desconocido")  # Mapear el valor de 'weather_id' a una etiqueta

        # Diccionario con las rutas manuales para cada tipo de clima
        image_paths = {
            "Soleado": "Image\Soleado.png",  # Reemplaza con la ruta correcta
            "Tormenta": "Image\Tormenta.png",  # Reemplaza con la ruta correcta
            "Lluvia": "Image\Lluvia.png",  # Reemplaza con la ruta correcta
            "Nublado": "Image\Cloudy.png",  # Reemplaza con la ruta correcta
            "Niebla": "Image\Fog.png",  # Reemplaza con la ruta correcta
        }

        # Ajustar las imágenes manualmente según su tipo de clima
        if predicted_weather == "Soleado":
            imagen_path = image_paths["Soleado"]
            st.image(imagen_path, width=320)
        elif predicted_weather == "Tormenta":
            imagen_path = image_paths["Tormenta"]
            st.image(imagen_path, width=320)
        elif predicted_weather == "Lluvia":
            imagen_path = image_paths["Lluvia"]
            st.image(imagen_path, width=320)
        elif predicted_weather == "Nublado":
            imagen_path = image_paths["Nublado"]
            st.image(imagen_path, width=310)
        elif predicted_weather == "Niebla":
            imagen_path = image_paths["Niebla"]
            st.image(imagen_path, width=320)


        # Mostrar un desplegable con la "Información de la predicción"
        with st.expander("Información de la predicción"):

            # Mostrar la predicción del clima
            st.write(f"**Predicción del clima:** {predicted_weather}")

            # Mostrar la precisión del modelo en el conjunto de prueba
            y_pred = best_model.predict(X_test)  # Predicciones en el conjunto de prueba
            accuracy = accuracy_score(y_test, y_pred)  # Calcular la precisión
            st.write(f"**Precisión en el conjunto de prueba:** {accuracy:.2f}")
            
            # Mostrar el R²
            r2 = r2_score(y_test, y_pred)  # Calcular R²
            st.write(f"**R² del modelo:** {r2:.2f}")
        
        # Mostrar el reporte de clasificación
        # st.write("**Reporte de clasificación:**")
        # report = classification_report(y_test, y_pred, target_names=weather_map.values(), output_dict=True)
        # st.write(report)