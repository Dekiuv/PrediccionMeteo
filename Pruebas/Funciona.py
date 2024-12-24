import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sqlite3

class SimuladorMeteorologico:
    def __init__(self, db_path):
        """
        Inicializa el simulador con la base de datos.
        :param db_path: Ruta a la base de datos SQLite.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3)

        # Diccionario para mapear números a nombres de condiciones de nubosidad
        self.cloudiness_mapping = {
            1: "parcialmente nublado",
            2: "despejado",
            3: "cubierto"
        }

        # Diccionario para mapear números a nombres de condiciones del clima
        self.weather_mapping = {
            1: "storm",
            2: "rain",
            3: "cloudy",
            4: "fog",
            5: "sun"
        }

    def cargar_datos_csv(self, csv_path, table_name):
        """
        Carga datos desde un archivo CSV y los almacena en la base de datos.
        :param csv_path: Ruta al archivo CSV.
        :param table_name: Nombre de la tabla donde se almacenarán los datos.
        """
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, self.connection, if_exists='replace', index=False)
        print(f"Datos cargados desde {csv_path} a la tabla {table_name}.")

    def procesar_datos_con_nubosidad_correcta(self):
        """
        Procesa los datos meteorológicos desde la tabla ValoresLimpios y maneja NaN directamente desde la base de datos.
        Además, se asegura de que la columna 'nubosidad_media' se mapee correctamente.
        """
        # Consultas de las tablas necesarias
        query_valores_limpios = "SELECT * FROM ValoresLimpios WHERE temp_max IS NOT NULL AND temp_min IS NOT NULL AND precipitation IS NOT NULL"
        query_dates = "SELECT * FROM dates"
        query_estaciones = "SELECT * FROM seasons"
        query_cloudiness = "SELECT * FROM cloudiness"
        query_weather = "SELECT * FROM weather"  # Para obtener el tiempo (weather)
        
        # Cargar los datos directamente desde la base de datos
        df_valores = pd.read_sql(query_valores_limpios, self.connection)
        df_dates = pd.read_sql(query_dates, self.connection)
        df_estaciones = pd.read_sql(query_estaciones, self.connection)
        df_cloudiness = pd.read_sql(query_cloudiness, self.connection)
        df_weather = pd.read_sql(query_weather, self.connection)

        # Unir las tablas de forma controlada para evitar NaN
        df = df_valores.merge(df_dates, on="date_id", how="left").merge(df_estaciones, on="estacion_id", how="left").merge(df_cloudiness, on="cloudiness_id", how="left").merge(df_weather, on="weather_id", how="left")

        # Verificar que la columna de nubosidad está correctamente nombrada
        if 'cloudiness' in df.columns:
            df.rename(columns={'cloudiness': 'nubosidad_media'}, inplace=True)

        # Eliminar filas con NaN en columnas críticas después de las uniones
        df.dropna(subset=["temp_max", "temp_min", "precipitation", "weather", "nubosidad_media"], inplace=True)

        # Convertir las fechas si están en formato numérico
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Filtrar fechas razonables
        df = df[(df['date'] >= '2000-01-01') & (df['date'] <= '2023-12-31')]

        # Codificar la columna 'weather' (categoría) con LabelEncoder
        le = LabelEncoder()
        df['weather'] = le.fit_transform(df['weather'])

        # Calcular la diferencia de temperatura
        df['temp_range'] = df['temp_max'] - df['temp_min']  # Diferencia entre temp_max y temp_min

        # Eliminar cualquier fila con NaN después de los cálculos
        df.dropna(subset=["temp_range"], inplace=True)

        # Ordenar por fecha
        df.sort_values(by='date', inplace=True)
        return df

    def mapear_nubosidad(self, df):
        """
        Mapea los valores de nubosidad de tipo categórico a numérico.
        """
        cloudiness_mapping = {
            "despejado": 2,
            "parcialmente nublado": 1,
            "cubierto": 3
        }
        df['nubosidad_media'] = df['nubosidad_media'].map(cloudiness_mapping)
        return df

    def entrenar_y_predecir(self, df, feature_columns, target_columns):
        """
        Entrena el modelo de predicción basado en los datos procesados y realiza predicciones para múltiples objetivos.
        """
        X = df[feature_columns].values
        y = df[target_columns].values  # Ahora 'y' es un DataFrame con múltiples columnas de objetivos

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizar características
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Entrenar el modelo de RandomForest para múltiples objetivos
        self.model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Evaluación de rendimiento para cada objetivo
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Imprimir resultados
        print(f"Entrenamiento - MAE: {mae_train:.2f}, MSE: {mse_train:.2f}")
        print(f"Prueba - MAE: {mae_test:.2f}, MSE: {mse_test:.2f}")

        # Mapear las predicciones numéricas a sus nombres
        y_pred_test_mapped = pd.DataFrame(y_pred_test, columns=target_columns)

        # Mapear las predicciones numéricas dentro del rango válido
        y_pred_test_mapped['weather'] = y_pred_test_mapped['weather'].apply(lambda x: self.weather_mapping.get(max(1, min(5, int(x))), 'Unknown'))
        y_pred_test_mapped['nubosidad_media'] = y_pred_test_mapped['nubosidad_media'].apply(lambda x: self.cloudiness_mapping.get(max(1, min(3, int(x))), 'Unknown'))

        # Guardar predicciones en un archivo CSV
        y_pred_test_mapped.to_csv("predicciones.csv", index=False)
        print("Predicciones guardadas en predicciones.csv")

        return y_pred_test_mapped, mae_test, mse_test


# Ejemplo de uso
if __name__ == "__main__":
    db_path = "CSV/Prediccion.db"  # Ruta a tu base de datos SQLite
    simulador = SimuladorMeteorologico(db_path)

    # Procesar los datos con nubosidad correctamente mapeada
    datos_procesados_con_nubosidad = simulador.procesar_datos_con_nubosidad_correcta()

    # Mapear la nubosidad a valores numéricos
    datos_procesados_con_nubosidad_mapeada = simulador.mapear_nubosidad(datos_procesados_con_nubosidad)

    # Entrenar el modelo y realizar predicciones para múltiples objetivos
    target_columns = ["temp_min", "temp_max", "precipitation", "weather", "nubosidad_media"]
    y_pred, mae, mse = simulador.entrenar_y_predecir(
        datos_procesados_con_nubosidad_mapeada,
        feature_columns=["temp_min", "humidity", "pressure", "solar_radiation", "precipitation", "nubosidad_media", "temp_range"],
        target_columns=target_columns
    )

    print(f"Predicciones: {y_pred[:5]}")
    print(f"MAE en prueba: {mae}")
    print(f"MSE en prueba: {mse}")