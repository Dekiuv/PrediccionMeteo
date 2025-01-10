import os
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend

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
def optimizar_y_predecir(df, subset_size=5000):
    feature_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']  # Características
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

    # Definir el modelo base
    model = SVC()

    # Definir el rango de hiperparámetros para optimizar
    param_distributions = {
        'kernel': ['rbf', 'linear', 'poly'],  
        'C': [0.1, 1, 10, 100],      
        'gamma': ['scale', 0.01, 0.1, 1]
    }

    # Configurar RandomizedSearchCV y guardar los resultados en un archivo
    with parallel_backend('threading'):
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, scoring='accuracy', verbose=2, n_jobs=-1, random_state=42)
        random_search.fit(X_train_scaled, y_train_resampled)

    # Guardar los resultados del ajuste en un archivo de texto en formato limpio
    with open("hyperparameter_tuning_log.txt", "w") as f:
        f.write("Hyperparameter Tuning Results\n")
        f.write("==========================\n")
        for i, params in enumerate(random_search.cv_results_['params']):
            f.write(f"Iteration {i+1}:\n")
            f.write(f"  Parameters: {params}\n")
            f.write(f"  Mean Fit Time: {random_search.cv_results_['mean_fit_time'][i]:.2f}s\n")
            f.write(f"  Mean Test Score: {random_search.cv_results_['mean_test_score'][i]:.4f}\n")
            f.write(f"  Std Test Score: {random_search.cv_results_['std_test_score'][i]:.4f}\n")
            f.write("--------------------------\n")

    # Mejor modelo encontrado
    best_model = random_search.best_estimator_
    st.write(f"Mejores hiperparámetros: {random_search.best_params_}")

    # Entrenar y predecir con el mejor modelo
    y_pred = best_model.predict(X_test_scaled)

    # Calcular Accuracy en porcentaje
    accuracy = accuracy_score(y_test, y_pred) * 100
    st.write(f"**Accuracy:** {accuracy:.2f}%")

    # Generar reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return report_df, best_model

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico - Optimización de Hiperparámetros")
st.write("Esta aplicación permite visualizar las predicciones del modelo meteorológico optimizado con SVC.")

# Cargar los datos
df_valores = cargar_datos()

st.write("Datos cargados correctamente:")
st.dataframe(df_valores.head(10))  # Muestra las primeras 10 filas de los datos

# Entrenar y predecir con optimización de hiperparámetros
if st.button("Optimizar y predecir"):
    st.write("Optimizando hiperparámetros y entrenando el modelo...")
    report_df, best_model = optimizar_y_predecir(df_valores)

    # Mostrar el reporte de clasificación
    st.write("**Reporte de Clasificación:**")
    st.dataframe(report_df)

    # Realizar una predicción con el mejor modelo
    st.write("Realizando la primera predicción...")
    sample = df_valores[['precipitation', 'temp_max', 'temp_min', 'wind']].iloc[0].values.reshape(1, -1)
    prediction = best_model.predict(sample)
    st.write(f"Predicción para la primera fila: {prediction[0]}")