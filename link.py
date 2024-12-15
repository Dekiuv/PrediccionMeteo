import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

# Función para cargar datos desde la base de datos
def load_data_from_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df  # Devuelve el DataFrame

# Función principal para entrenar SVM y hacer predicciones
def main():
    # Parámetros de la base de datos
    db_path = "CSV/observations.db"  # Ruta de la base de datos
    table_name = "weather_data"      # Nombre de la tabla

    # Cargar los datos desde la base de datos
    print("Cargando los datos desde la base de datos...")
    try:
        df = load_data_from_db(db_path, table_name)
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return

    # Verificar las columnas necesarias
    required_columns = ['precipitation', 'temp_max', 'temp_min']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: La tabla debe contener las columnas {required_columns}")
        return

    # Calcular probabilidad de precipitación
    precip_threshold = 50  # Umbral máximo de precipitación en mm
    df['precip_probability'] = (df['precipitation'] / precip_threshold) * 100
    df['precip_probability'] = df['precip_probability'].clip(0, 100)  # Limitar al 100%

    # Seleccionar características y objetivos
    X = df['precipitation'].values.reshape(-1, 1)  # Predictor: 'precipitation'
    y_max = df['temp_max'].values  # Objetivo: temperatura máxima
    y_min = df['temp_min'].values  # Objetivo: temperatura mínima
    y_prob = df['precip_probability'].values  # Objetivo: % de precipitación

    # Normalizar los datos
    scaler_X = MinMaxScaler()
    scaler_y_max = MinMaxScaler()
    scaler_y_min = MinMaxScaler()
    scaler_y_prob = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_max_scaled = scaler_y_max.fit_transform(y_max.reshape(-1, 1)).ravel()
    y_min_scaled = scaler_y_min.fit_transform(y_min.reshape(-1, 1)).ravel()
    y_prob_scaled = scaler_y_prob.fit_transform(y_prob.reshape(-1, 1)).ravel()

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_max_train, y_max_test = train_test_split(X_scaled, y_max_scaled, test_size=0.2, random_state=42)
    _, _, y_min_train, y_min_test = train_test_split(X_scaled, y_min_scaled, test_size=0.2, random_state=42)
    _, _, y_prob_train, y_prob_test = train_test_split(X_scaled, y_prob_scaled, test_size=0.2, random_state=42)

    # Crear y entrenar tres modelos SVM
    model_max = SVR(kernel='rbf', C=10, gamma='auto')
    model_min = SVR(kernel='rbf', C=10, gamma='auto')
    model_prob = SVR(kernel='rbf', C=10, gamma='auto')
    model_max.fit(X_train, y_max_train)
    model_min.fit(X_train, y_min_train)
    model_prob.fit(X_train, y_prob_train)

    # ------------------------
    # Predicción para 7 días futuros
    # ------------------------
    print("Predicción de los próximos 7 días:")
    future_precipitation = np.linspace(X.max(), X.max() + 7, 7).reshape(-1, 1)  # Generar 7 días ficticios de precipitación
    future_precip_scaled = scaler_X.transform(future_precipitation)

    # Predicción de temperaturas máximas, mínimas y % de precipitación
    future_temp_max_scaled = model_max.predict(future_precip_scaled)
    future_temp_min_scaled = model_min.predict(future_precip_scaled)
    future_prob_scaled = model_prob.predict(future_precip_scaled)

    future_temp_max = scaler_y_max.inverse_transform(future_temp_max_scaled.reshape(-1, 1))
    future_temp_min = scaler_y_min.inverse_transform(future_temp_min_scaled.reshape(-1, 1))
    future_prob = scaler_y_prob.inverse_transform(future_prob_scaled.reshape(-1, 1))

    # Mostrar las predicciones
    print("\nResultados:")
    for i, (temp_max, temp_min, prob) in enumerate(zip(future_temp_max, future_temp_min, future_prob), 1):
        print(f"Día {i}: Temperatura Máxima = {temp_max[0]:.2f}°C, Temperatura Mínima = {temp_min[0]:.2f}°C, "
              f"Probabilidad de Precipitación = {prob[0]:.2f}%")

    # Visualización de las predicciones futuras
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 8), future_temp_max, marker='o', linestyle='-', label='Temperatura Máxima', color='r')
    plt.plot(range(1, 8), future_temp_min, marker='o', linestyle='-', label='Temperatura Mínima', color='b')
    plt.plot(range(1, 8), future_prob, marker='o', linestyle='--', label='% Precipitación', color='g')
    plt.xlabel('Días Futuros')
    plt.ylabel('Valores')
    plt.title('Predicción de Temperatura Máxima, Mínima y % de Precipitación')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
