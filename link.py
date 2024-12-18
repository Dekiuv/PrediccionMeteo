import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Función para cargar datos desde la base de datos
def load_data_from_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Título de la aplicación
st.title("Predicción Meteorológica Mejorada con Variables Balanceadas")

# Parámetros de entrada
st.sidebar.header("Parámetros de Entrada")
db_path = st.sidebar.text_input("Ruta de la base de datos", "CSV/observations.db")
table_name = st.sidebar.text_input("Nombre de la tabla", "weather_data")

# Botón para cargar los datos
if st.sidebar.button("Cargar Datos"):
    try:
        # Cargar datos
        df = load_data_from_db(db_path, table_name)
        st.write("### Datos Cargados")
        st.write(df.head())

        # Verificar distribución de valores categóricos
        st.write("### Distribución de Clases Categóricas")
        st.write(df['weather_id'].value_counts())

        # Transformar variables categóricas con One-Hot Encoding
        df_encoded = pd.get_dummies(df, columns=['weather_id', 'cloudiness_id', 'estacion_id'], drop_first=True)

        # Variables predictoras y objetivo
        features = ['precipitation', 'temp_min', 'wind', 'humidity', 'pressure', 'solar_radiation', 'visibility']
        features += [col for col in df_encoded.columns if 'weather_id' in col or 'cloudiness_id' in col or 'estacion_id' in col]
        target = 'temp_max'

        X = df_encoded[features]
        y = df_encoded[target]

        # Normalizar datos
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        # Entrenar modelos
        st.write("### Comparación de Modelos")

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        # XGBoost
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # Desnormalizar para métricas y visualización
        y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        rf_pred_real = scaler_y.inverse_transform(rf_pred.reshape(-1, 1)).ravel()
        xgb_pred_real = scaler_y.inverse_transform(xgb_pred.reshape(-1, 1)).ravel()

        # Función para mostrar métricas
        def show_metrics(model_name, y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            st.write(f"**{model_name}**")
            st.write(f"- MAE: {mae:.4f}")
            st.write(f"- RMSE: {rmse:.4f}")
            st.write(f"- R²: {r2:.4f}")

        # Mostrar métricas
        show_metrics("Random Forest", y_test_real, rf_pred_real)
        show_metrics("XGBoost", y_test_real, xgb_pred_real)

        # Gráfico comparativo
        st.write("### Comparación de Valores Reales y Predichos")
        fig, ax = plt.subplots()
        ax.plot(y_test_real, label="Valores Reales", color="blue")
        ax.plot(rf_pred_real, label="Random Forest", color="green", linestyle="--")
        ax.plot(xgb_pred_real, label="XGBoost", color="red", linestyle="--")
        ax.set_title("Valores Reales vs Predichos")
        ax.set_xlabel("Muestras")
        ax.set_ylabel("Temperatura Máxima (°C)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
