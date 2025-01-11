import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend
from sklearn.feature_selection import f_classif
from sklearn.metrics import r2_score

# Configuración cabezera de la página
st.set_page_config(
    page_title="Simulador Meteorológico",  # Título de la página en el navegador
    page_icon="🌤️",  # Icono de la página
    layout="wide"  # Diseño de la página (en este caso, "ancho")
)

# Configurar OMP_NUM_THREADS para optimizar el uso de CPU
os.environ["OMP_NUM_THREADS"] = "4"

# Función para cargar datos desde SQLite
@st.cache_data
def cargar_datos():
    db_path = "CSV/Prediccion.db"  # Cambiar a la ruta correcta
    connection = sqlite3.connect(db_path)
    query_valores = "SELECT * FROM ValoresLimpios"
    
    df_valores = pd.read_sql_query(query_valores, connection)
    connection.close()
    return df_valores

# Función para optimizar hiperparámetros y entrenar el modelo con GridSearchCV
def optimizar_y_predecir(df, subset_size=5000):
    feature_columns = ['date_id', 'precipitation','temp_max','wind','humidity', 'cloudiness_id']  # Características
    target_column = 'weather_id'

    X = df[feature_columns]
    y = df[target_column]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Usar un subconjunto de datos para la búsqueda
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_subset, y_train_subset)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Definir el espacio de hiperparámetros para GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly']
    }

    # Configurar GridSearchCV
    grid_search = GridSearchCV(SVC(), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train_resampled)

    # Mejor modelo encontrado
    best_model = grid_search.best_estimator_
    st.write(f"Mejores hiperparámetros: {grid_search.best_params_}")

    # Entrenar y predecir con el mejor modelo
    y_pred = best_model.predict(X_test_scaled)

    # Calcular Accuracy en porcentaje
    accuracy = accuracy_score(y_test, y_pred) * 100
    st.write(f"**Accuracy:** {accuracy:.2f}%")

    # Generar reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return report_df, best_model, X, y

# Función para realizar el análisis de varianza (ANOVA)
def anova_analysis(X, y):
    # ANOVA: Comparar la varianza de las características con respecto a las clases
    f_values, p_values = f_classif(X, y)
    st.write("Resultados del ANOVA (valor F y valor p para cada característica):")
    feature_names = X.columns
    anova_results = pd.DataFrame({
        'Feature': feature_names,
        'F-Value': f_values,
        'P-Value': p_values
    })
    st.dataframe(anova_results)

    return anova_results

# Función para calcular el R² (coeficiente de determinación)
def calculate_r2(model, X, y):
    # Calcular R² usando el método score de SVC
    r2 = model.score(X, y)
    st.write(f"Coeficiente de determinación R²: {r2:.2f}")
    return r2

# Diccionario de mapeo para 'weather_id'
weather_map = {
    1: "Tormenta",
    2: "Lluvia",
    3: "Nublado",
    4: "Niebla",
    5: "Soleado"
}

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico")
st.write("Esta aplicación permite visualizar las predicciones del modelo meteorológico optimizado con SVC.")

# Cargar los datos
df_valores = cargar_datos()

# Permitir al usuario seleccionar los valores para la predicción
st.write("Ingrese los valores para realizar la predicción:")

date_id = st.number_input('ID de la fecha', min_value=1, max_value=10000, value=1, step=1)
precipitation = st.number_input('Precipitación (mm)', min_value=0.0, max_value=500.0, value=0.0, step=0.1)
temp_max = st.number_input('Temperatura máxima (°C)', min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
wind = st.number_input('Viento (km/h)', min_value=0.0, max_value=150.0, value=10.0, step=0.1)
humidity = st.number_input('Humedad (%)', min_value=0, max_value=100, value=50, step=1)
cloudiness_id = st.number_input('Índice de nubosidad', min_value=0, max_value=10, value=5, step=1)


# Entrenar y predecir con optimización de hiperparámetros
if st.button("Optimizar y predecir"):
    st.write("Optimizando hiperparámetros y entrenando el modelo...")
    report_df, best_model, X, y = optimizar_y_predecir(df_valores)

    # Mostrar el reporte de clasificación
    st.write("**Reporte de Clasificación:**")
    st.dataframe(report_df)

    # Realizar el análisis de varianza en las características
    anova_results = anova_analysis(X, y)

    # Calcular el coeficiente de determinación R²
    r2 = calculate_r2(best_model, X, y)

    # Realizar una predicción con los valores ingresados
    st.write("Realizando la predicción con los valores ingresados...")
    sample = [[date_id, precipitation, temp_max, wind, humidity, cloudiness_id]]
    prediction = best_model.predict(sample)

    # Convertir la predicción en una descripción legible
    predicted_weather = weather_map.get(prediction[0], "Desconocido")
    st.write(f"La predicción del clima es: {predicted_weather}")
