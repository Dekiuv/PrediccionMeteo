import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Configurar OMP_NUM_THREADS para optimizar el uso de CPU
os.environ["OMP_NUM_THREADS"] = "4"

# Configuración de Streamlit
st.set_page_config(
    page_title="Simulador Meteorológico",
    page_icon="🌤️",
    layout="centered"
)

# Definir el estado de la página en el session_state si no está definido
if 'page' not in st.session_state:
    st.session_state.page = 'bienvenida'  # Página inicial

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
    target_column = 'weather_id'

    X = df[feature_columns]
    y = df[target_column]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler

# Función para optimizar hiperparámetros y entrenar el modelo
def optimizar_y_entrenar(X_train, y_train):
    param_distributions = {
        'C': [10],
        'gamma': [0.1],
        'kernel': ['rbf']
    }

    # Optimizar modelo con RandomizedSearchCV
    model = SVC(probability=True)  # Activar las probabilidades
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)

    st.write(f"**Mejores hiperparámetros:** {random_search.best_params_}")
    
    # Guardar los resultados de la búsqueda de hiperparámetros en un archivo de texto en la misma carpeta
    archivo_guardado = 'resultados_hiperparametros.txt'  # El archivo se guarda automáticamente en la carpeta actual
    with open(archivo_guardado, 'w') as f:
        f.write("Hyperparameter Tuning Results\n")
        f.write("==========================\n")
        for i, result in enumerate(random_search.cv_results_['params']):
            f.write(f"Iteration {i+1}:\n")
            f.write(f"  Parameters: {result}\n")
            f.write(f"  Mean Fit Time: {random_search.cv_results_['mean_fit_time'][i]:.2f}s\n")
            f.write(f"  Mean Test Score: {random_search.cv_results_['mean_test_score'][i]:.4f}\n")
            f.write(f"  Std Test Score: {random_search.cv_results_['std_test_score'][i]:.4f}\n")
            f.write("--------------------------\n")

    return random_search.best_estimator_, archivo_guardado, random_search

# Diccionario de mapeo para 'weather_id'
weather_map = {
    1: "Tormenta",
    2: "Lluvia",
    3: "Nublado",
    4: "Niebla",
    5: "Soleado"
}

# Lógica para mostrar diferentes páginas
if st.session_state.page == 'bienvenida':
    # Página de bienvenida
    st.title("Bienvenido al Simulador Meteorológico 🌤️")
    st.write("Esta aplicación permite predecir las condiciones meteorológicas basándose en datos históricos.")
    st.write("Haz clic en el botón para comenzar con el simulador.")

    # Botón para ir a la página del simulador
    if st.button("Ir al simulador"):
        st.session_state.page = 'simulador'  # Cambiar la página a simulador

elif st.session_state.page == 'simulador':
    # Cargar datos
    df_valores = cargar_datos()

    # Características fijas para el modelo
    features_options = ['date_id', 'precipitation', 'temp_max', 'temp_min', 'wind', 'humidity', 'pressure', 'solar_radiation', 'visibility', 'cloudiness_id', 'estacion_id']
    selected_features = features_options  # Usamos estas características siempre

    # Preparar los datos con las características seleccionadas
    X_train, X_test, y_train, y_test, scaler = preparar_datos(df_valores, selected_features)

    # Entrenar y optimizar el modelo
    best_model = None  # Inicializamos el modelo en None

    st.title("Simulador Meteorológico - Predicción con SVC")
    st.write("Esta aplicación permite predecir las condiciones meteorológicas basándose en los datos ingresados.")

    # Formulario de predicción
    date_id = st.number_input('ID de fecha', min_value=0, max_value=100000, value=1)
    precipitation = st.number_input('Precipitación (mm)', min_value=0.0, max_value=500.0, value=0.0, step=0.1)
    temp_max = st.number_input('Temperatura máxima (°C)', min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    temp_min = st.number_input('Temperatura mínima (°C)', min_value=-10.0, max_value=50.0, value=15.0, step=0.1)
    wind = st.number_input('Viento (km/h)', min_value=0.0, max_value=150.0, value=10.0, step=0.1)
    humidity = st.number_input('Humedad (%)', min_value=0, max_value=100, value=60, step=1)
    pressure = st.number_input('Presión atmosférica (hPa)', min_value=900, max_value=1100, value=1015, step=1)
    solar_radiation = st.number_input('Radiación solar (W/m²)', min_value=0, max_value=2000, value=500, step=1)
    visibility = st.number_input('Visibilidad (km)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    cloudiness_id = st.number_input('Nubosidad (1: Parcialmente nublado, 2: Despejado, 3: Cubierto)', min_value=1, max_value=3, value=1, step=1)
    estacion_id = st.number_input('Estación (1: Invierno, 2: Primavera, 3: Verano, 4: Otoño)', min_value=1, max_value=4, value=1, step=1)

    # Botón para hacer la predicción
    if st.button("Hacer predicción"):
        with st.spinner('Entrenando el modelo y optimizando hiperparámetros...'):
            best_model, archivo_guardado, random_search = optimizar_y_entrenar(X_train, y_train)
        
        st.success("Entrenamiento completado!")

        # Predicción con los valores del usuario
        sample = scaler.transform([[date_id, precipitation, temp_max, temp_min, wind, humidity, pressure, solar_radiation, visibility, cloudiness_id, estacion_id]])
        prediction = best_model.predict(sample)
        predicted_weather = weather_map.get(prediction[0], "Desconocido")
        
        st.write(f"**Predicción del clima:** {predicted_weather}")

        # Mostrar la precisión del modelo en el conjunto de prueba
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Precisión en el conjunto de prueba:** {accuracy:.2f}")

        # Mostrar la ubicación donde se guardó el archivo
        st.write(f"**Resultados de la búsqueda de hiperparámetros guardados en:** {archivo_guardado}")
    
    # Botón para volver a la página de bienvenida
    if st.button("Volver a la bienvenida"):
        st.session_state.page = 'bienvenida'  # Volver a la página de bienvenida
