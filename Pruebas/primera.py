import sqlite3
import pandas as pd
import streamlit as st
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración de visualización
sns.set(font_scale=1.3)
sns.set_style('whitegrid')

# Función para cargar datos
@st.cache_data
def load_data_from_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Preprocesamiento de datos
def preprocess_data(df, predictors, target):
    # Imputar valores faltantes
    imputer = SimpleImputer(strategy="mean")
    df[predictors] = imputer.fit_transform(df[predictors])

    # Feature Engineering: Crear nuevas características
    df['temp_diff'] = df['temp_max'] - df['temp_min']  # Diferencia entre temp_max y temp_min
    predictors.append('temp_diff')

    # Transformar variables con valores extremos
    transformer = FunctionTransformer(np.log1p, validate=True)  # Log(1 + x)
    for col in ['precipitation', 'wind']:
        df[col] = transformer.transform(df[[col]])

    # Escalado de características
    scaler = StandardScaler()
    X = scaler.fit_transform(df[predictors])
    y = df[target]

    return X, y, scaler

# Visualización de correlación
def plot_correlation(df, predictors, target):
    st.subheader("Matriz de Correlación")
    corr_matrix = df[predictors + [target]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlación entre variables")
    st.pyplot(fig)

# Entrenamiento y evaluación del modelo
def train_and_evaluate(X, y, model_type="SVR"):
    # Dividir datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "SVR":
        # Optimización de hiperparámetros para SVM
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        }
        model = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    elif model_type == "RandomForest":
        # Hiperparámetros para Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    elif model_type == "XGBoost":
        # Hiperparámetros para XGBoost
        param_grid = {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'subsample': [0.7, 0.8, 1]
        }
        model = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    model.fit(X_train, y_train)

    # Mejor modelo
    best_model = model.best_estimator_
    best_params = model.best_params_

    # Validación cruzada
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="neg_mean_squared_error")

    # Evaluación
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return best_model, predictions, X_test, y_test, mse, mae, r2, best_params, -cv_scores.mean()

# Visualización de métricas
def plot_metrics(y_test, predictions):
    st.subheader("Distribución de Errores")
    errors = predictions - y_test
    fig, ax = plt.subplots()
    sns.histplot(errors, bins=30, kde=True, ax=ax, color='skyblue')
    ax.set_title("Distribución de Errores")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

    st.subheader("Valores Reales vs Predicciones")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=predictions, ax=ax, alpha=0.7, edgecolor='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    ax.set_title("Valores Reales vs Predicciones")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Predicciones")
    st.pyplot(fig)

# Predicción del tiempo (lluvia, nublado, soleado)
def predict_weather_conditions(model, recent_data, scaler, days=7):
    st.subheader("Pronóstico para los próximos días")
    conditions = []
    current_data = scaler.transform(recent_data)

    for day in range(days):
        temp_prediction = model.predict(current_data)[0]
        precipitation = np.random.uniform(0, 100)  # Porcentaje de probabilidad de lluvia
        if precipitation < 20:
            condition = "Soleado"
        elif 20 <= precipitation < 60:
            condition = "Nublado"
        else:
            condition = "Lluvioso"

        conditions.append((temp_prediction, precipitation, condition))

        # Simular el siguiente día (ajustar datos aleatoriamente)
        current_data[0, 0] = np.random.uniform(0, 100)  # Actualizar precipitación aleatoria
        current_data[0, 1] = temp_prediction + np.random.uniform(-2, 2)  # Variar la temperatura

    # Mostrar predicciones
    for i, (temp, prec, cond) in enumerate(conditions, 1):
        st.write(f"Día {i}: Temperatura Máxima: {temp:.2f}°C, Probabilidad de Lluvia: {prec:.2f}%, Condición: {cond}")

# Interfaz con Streamlit
def main():
    st.title("Simulador de Predicción Meteorológica")

    # Ruta a la base de datos y nombre de la tabla
    db_path = "CSV/observations.db"
    table_name = "weather_data"

    # Cargar los datos
    st.write("Cargando los datos desde la base de datos...")
    df = load_data_from_db(db_path, table_name)

    # Selección de variables predictoras y objetivo
    predictors = ['precipitation', 'temp_min', 'wind', 'humidity', 'pressure', 'solar_radiation']
    target = 'temp_max'

    # Visualizar correlación
    plot_correlation(df, predictors, target)

    # Preprocesamiento
    X_scaled, y, scaler = preprocess_data(df, predictors, target)

    # Entrenar modelo
    st.sidebar.subheader("Entrenar Modelo")
    model_type = st.sidebar.selectbox("Seleccionar modelo", ["SVR", "RandomForest", "XGBoost"])
    if st.sidebar.button("Entrenar"):
        model, predictions, X_test, y_test, mse, mae, r2, best_params, mean_cv_mse = train_and_evaluate(X_scaled, y, model_type)
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.write(f"Error Cuadrático Medio (MSE): {mse}")
        st.write(f"Error Absoluto Medio (MAE): {mae}")
        st.write(f"Coeficiente de Determinación (R²): {r2}")
        st.write(f"Mejores Hiperparámetros: {best_params}")
        st.write(f"Validación Cruzada (MSE Promedio): {mean_cv_mse}")
        plot_metrics(y_test, predictions)

    # Predicción futura
    st.sidebar.subheader("Pronóstico Futuro")
    days_to_forecast = st.sidebar.slider("Días a predecir", 1, 14, 7)
    if st.sidebar.button("Pronosticar"):
        if "model" in st.session_state and "scaler" in st.session_state:
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            recent_data = df[predictors].tail(1)  # Usar el último día como base
            predict_weather_conditions(model, recent_data, scaler, days_to_forecast)
        else:
            st.warning("Por favor, entrena el modelo antes de realizar un pronóstico.")

if __name__ == "__main__":
    main()