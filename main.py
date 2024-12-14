import sqlite3
import pandas as pd
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Función para cargar datos
@st.cache_data
def load_data_from_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Función para entrenar el modelo

def train_and_evaluate(df):
    predictors = ['precipitation', 'temp_min', 'wind', 'humidity', 'pressure', 'solar_radiation']
    target = 'temp_max'

    X = df[predictors]
    y = df[target]

    # Escalado de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Optimización de hiperparámetros para SVM
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Mejor modelo
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Validación cruzada
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring="neg_mean_squared_error")

    # Entrenar y predecir
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return best_model, predictions, X_test, y_test, mse, mae, r2, best_params, -cv_scores.mean()

# Visualización de gráficos
def plot_metrics(y_test, predictions):
    st.subheader("Distribución de Errores")
    errors = predictions - y_test
    fig, ax = plt.subplots()
    ax.hist(errors, bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Distribución de Errores (Predicción - Real)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

    st.subheader("Valores Reales vs Predicciones")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.7, color='blue', edgecolor='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    ax.set_title("Valores Reales vs Predicciones")
    ax.set_xlabel("Valores Reales")
    ax.set_ylabel("Predicciones")
    st.pyplot(fig)

# Predicción futura
def predict_future(model, recent_data, scaler, days=30):
    future_predictions = []
    current_data = scaler.transform(recent_data[-1:])

    for _ in range(days):
        prediction_temp = model.predict(current_data)[0]
        precipitation = np.random.uniform(0, 100)
        weather = "Soleado" if precipitation < 30 else "Lluvioso"

        future_predictions.append({
            "temp": prediction_temp,
            "precipitation": precipitation,
            "weather": weather
        })

        # Actualizar datos
        current_data = current_data.copy()
        current_data[0, 0] = precipitation
        current_data[0, 1] = prediction_temp
        current_data[0, 2:] = np.random.uniform(0, 100, size=(1, 4))

    return future_predictions

# Interfaz con Streamlit
def main():
    st.title("Análisis de Datos Meteorológicos")

    # Ruta a la base de datos y nombre de la tabla
    db_path = "CSV/observations.db"
    table_name = "weather_data"

    # Cargar los datos
    st.write("Cargando los datos desde la base de datos...")
    df = load_data_from_db(db_path, table_name)

    # Mostrar los datos en Streamlit
    st.write("Datos cargados:")
    st.dataframe(df)

    # Entrenar modelo
    st.sidebar.subheader("Entrenar Modelo")
    if st.sidebar.button("Entrenar"):
        model, predictions, X_test, y_test, mse, mae, r2, best_params, mean_cv_mse = train_and_evaluate(df)
        st.session_state["model"] = model
        st.session_state["scaler"] = StandardScaler().fit(df[['precipitation', 'temp_min', 'wind', 'humidity', 'pressure', 'solar_radiation']])
        st.write(f"Error Cuadrático Medio (MSE): {mse}")
        st.write(f"Error Absoluto Medio (MAE): {mae}")
        st.write(f"Coeficiente de Determinación (R²): {r2}")
        st.write(f"Mejores Hiperparámetros: {best_params}")
        st.write(f"Validación Cruzada (MSE Promedio): {mean_cv_mse}")
        plot_metrics(y_test, predictions)

    # Predicción futura
    st.sidebar.subheader("Predicción Futura")
    days = st.sidebar.slider("Días a predecir", 1, 30, 7)
    if st.sidebar.button("Predecir"):
        if "model" in st.session_state and "scaler" in st.session_state:
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            recent_data = df[['precipitation', 'temp_min', 'wind', 'humidity', 'pressure', 'solar_radiation']].tail(7)
            future_predictions = predict_future(model, recent_data, scaler, days)
            st.write("Predicciones futuras:")
            for day in future_predictions:
                st.write(f"Temperatura: {day['temp']:.2f}°C, Precipitación: {day['precipitation']:.2f}%, Tiempo: {day['weather']}")
        else:
            st.warning("Por favor, entrena el modelo antes de predecir.")

if __name__ == "__main__":
    main()

#Ejecutar: python3 -m streamlit run main.py