import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

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

# Función para optimizar hiperparámetros y entrenar el modelo
def optimizar_y_predecir(df, tolerance=0.2, subset_size=5000):
    feature_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']  # Características
    target_column = 'weather_id'

    X = df[feature_columns]
    y = df[target_column]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Usar un subconjunto de datos para la búsqueda
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_subset, y_train_subset)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Definir el modelo base
    model = SVR()

    # Definir el rango de hiperparámetros para optimizar
    param_distributions = {
        'kernel': ['rbf', 'linear'],  
        'C': [0.1, 1, 10, 100],      
        'gamma': ['scale', 0.01, 0.1, 1]
    }

    # Configurar RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, scoring='r2', verbose=2, n_jobs=-1, random_state=42)
    random_search.fit(X_train_scaled, y_train_resampled)

    # Mejor modelo encontrado
    best_model = random_search.best_estimator_
    st.write(f"Mejores hiperparámetros: {random_search.best_params_}")

    # Entrenar y predecir con el mejor modelo
    y_pred = best_model.predict(X_test_scaled)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calcular accuracy basado en un margen relativo
    correct_predictions = ((abs(y_test - y_pred) / y_test) <= tolerance).sum()
    accuracy = (correct_predictions / len(y_test)) * 100  # Accuracy en %

    # Crear un DataFrame con resultados
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return mae, mse, r2, accuracy, results

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico - Optimización de Hiperparámetros")
st.write("Esta aplicación permite visualizar las predicciones del modelo meteorológico optimizado con SVR.")

# Cargar los datos
df_valores = cargar_datos()

st.write("Datos cargados correctamente:")
st.dataframe(df_valores.head(10))  # Muestra las primeras 10 filas de los datos

# Entrenar y predecir con optimización de hiperparámetros
if st.button("Optimizar y predecir"):
    st.write("Optimizando hiperparámetros y entrenando el modelo...")
    mae, mse, r2, accuracy, results = optimizar_y_predecir(df_valores)

    # Mostrar métricas
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}%")

    # Mostrar resultados limitados a las primeras 10 predicciones
    st.write("**Resultados de las predicciones:**")
    st.dataframe(results.head(10))