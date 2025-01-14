import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

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
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': [class_weights]  # Usar los pesos de clase
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

# Cargar datos
df_valores = cargar_datos()

# Características fijas para el modelo
features_options = ['precipitation', 'wind', 'humidity', 'visibility']
selected_features = features_options  # Usamos estas características siempre

# Preparar los datos con las características seleccionadas
X_train, X_test, y_train, y_test, scaler = preparar_datos(df_valores, selected_features)

# Entrenar y optimizar el modelo
best_model = None  # Inicializamos el modelo con None

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico")
st.write("Esta aplicación permite predecir las condiciones meteorológicas basándose en los datos ingresados.")

# Formulario interactivo para que el usuario ingrese los datos
st.write("**Ingrese los valores para realizar una predicción personalizada:**")

precipitation = st.number_input('Precipitación (mm)', min_value=0.0, max_value=500.0, value=0.0, step=0.1)
wind = st.number_input('Viento (km/h)', min_value=0.0, max_value=150.0, value=10.0, step=0.1)
humidity = st.number_input('Humedad (%)', min_value=0, max_value=100, value=60, step=1)
visibility = st.number_input('Visibilidad (km)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# Botón para entrenar el modelo y hacer la predicción
if st.button("Entrenar y Predecir Clima"):
    with st.spinner('Entrenando el modelo y optimizando hiperparámetros...'):
        best_model, archivo_guardado, random_search = optimizar_y_entrenar(X_train, y_train)  # Entrenar y optimizar el modelo
    
    st.success("Entrenamiento completado!")

    # Predicción con los valores del usuario
    sample = scaler.transform([[precipitation, wind, humidity, visibility]])
    prediction = best_model.predict(sample)  # Realizar la predicción
    predicted_weather = weather_map.get(prediction[0], "Desconocido")  # Mapear el valor de 'weather_id' a una etiqueta
    
    st.write(f"**Predicción del clima:** {predicted_weather}")

    # Mostrar la precisión del modelo en el conjunto de prueba
    y_pred = best_model.predict(X_test)  # Predicciones en el conjunto de prueba
    accuracy = accuracy_score(y_test, y_pred)  # Calcular la precisión
    st.write(f"**Precisión en el conjunto de prueba:** {accuracy:.2f}")
    
    # Mostrar el R²
    r2 = r2_score(y_test, y_pred)  # Calcular R²
    st.write(f"**R² del modelo:** {r2:.2f}")
    
    # Mostrar el reporte de clasificación
    st.write("**Reporte de clasificación:**")
    report = classification_report(y_test, y_pred, target_names=weather_map.values(), output_dict=True)
    st.write(report)