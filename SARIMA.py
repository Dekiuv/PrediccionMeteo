import pandas as pd
import sqlite3
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SARIMAModel:
    def __init__(self, db_path):
        self.db_path = db_path
        self.label_encoder = LabelEncoder()  # Para codificar etiquetas de clima

    def cargar_datos(self):
        # Conectar a la base de datos
        connection = sqlite3.connect(self.db_path)
        query = """
        SELECT d.date, w.weather, vl.temp_max, vl.temp_min, vl.precipitation, vl.humidity, vl.wind
        FROM ValoresLimpios vl
        JOIN dates d ON vl.date_id = d.date_id
        JOIN weather w ON vl.weather_id = w.weather_id
        """
        df = pd.read_sql(query, connection)
        connection.close()
        
        # Convertir la columna de fecha a formato datetime
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)  # Ordenar por fecha
        
        return df

    def procesar_datos(self, df):
        # Codificar la columna "weather" como numérica
        df['weather_encoded'] = self.label_encoder.fit_transform(df['weather'])
        return df

    def entrenar_sarima(self, df, target_column, order, seasonal_order):
        """
        Entrenar el modelo SARIMA en una serie temporal específica.
        
        Args:
            df (pd.DataFrame): Datos procesados con columna temporal.
            target_column (str): Columna a predecir (por ejemplo, 'weather_encoded').
            order (tuple): Parámetros (p, d, q) del modelo SARIMA.
            seasonal_order (tuple): Parámetros estacionales (P, D, Q, s).

        Returns:
            Modelo SARIMA entrenado y su resumen.
        """
        # Seleccionar el objetivo
        y = df.set_index('date')[target_column]

        # Entrenar el modelo SARIMA
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        
        return results

    def predecir(self, modelo, pasos):
        """
        Realizar predicciones con un modelo SARIMA entrenado.
        
        Args:
            modelo: Modelo SARIMA entrenado.
            pasos (int): Número de pasos futuros a predecir.

        Returns:
            pd.DataFrame: Predicciones en un DataFrame.
        """
        pred = modelo.get_forecast(steps=pasos)
        pred_ci = pred.conf_int()
        forecast = pred.predicted_mean
        
        return pd.DataFrame({
            "Forecast": forecast,
            "Lower Bound": pred_ci.iloc[:, 0],
            "Upper Bound": pred_ci.iloc[:, 1]
        })

    def descodificar_weather(self, predicciones):
        """
        Descodificar valores predichos de 'weather_encoded' a las etiquetas originales.
        
        Args:
            predicciones: DataFrame con las predicciones de clima en formato codificado.

        Returns:
            DataFrame con las etiquetas originales.
        """
        predicciones['Weather'] = self.label_encoder.inverse_transform(predicciones['Forecast'].round().astype(int))
        return predicciones
