import pandas as pd # Librería para manejo de datos
import sqlite3 # Librería para manejo de bases de datos
from sklearn.ensemble import RandomForestRegressor # Modelo de regresión
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Métricas de evaluación
from sklearn.model_selection import train_test_split # División de datos
from sklearn.preprocessing import StandardScaler, LabelEncoder # Preprocesamiento de datos

# Clase para el simulador meteorológico
class SimuladorMeteorologico:
    def __init__(self, db_path):
        # Definición del constructor
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3)

        # Mapeo de los valores de clima y nubosidad
        self.weather_mapping = {
            1: "storm",
            2: "rain",
            3: "cloudy",
            4: "fog",
            5: "sun"
        }
        # Mapeo de los valores de nubosidad
        self.cloudiness_mapping = {
            1: "parcialmente nublado",
            2: "cubierto",
            3: "despejado"
        }

    # Método para cargar los datos de la base de datos
    def cargar_datos(self):
        query = """
        SELECT vl.*, d.date, s.estacion, c.cloudiness AS cloudiness, w.weather
        FROM ValoresLimpios vl
        JOIN dates d ON vl.date_id = d.date_id
        JOIN seasons s ON vl.estacion_id = s.estacion_id
        JOIN cloudiness c ON vl.cloudiness_id = c.cloudiness_id
        JOIN weather w ON vl.weather_id = w.weather_id
        """
        df = pd.read_sql(query, self.connection) # Cargar los datos de la base de datos
        df.drop(columns=['date_id', 'weather_id', 'cloudiness_id', 'estacion_id'], inplace=True) # Eliminar columnas innecesarias
        cols = ['date'] + [col for col in df.columns if col != 'date'] # Reordenar las columnas
        df = df[cols]
        return df

    # Método para procesar los datos
    def procesar_datos(self, df):
        le_weather = LabelEncoder() # Codificación de etiquetas para la columna de clima
        df['weather'] = le_weather.fit_transform(df['weather']) # Codificar las etiquetas de clima
        le_nubosidad = LabelEncoder() # Codificación de etiquetas para la columna de nubosidad
        df['cloudiness'] = le_nubosidad.fit_transform(df['cloudiness']) # Codificar las etiquetas de nubosidad
        df['temp_range'] = df['temp_max'] - df['temp_min'] # Calcular el rango de temperatura
        return df
    
    # Método para entrenar y predecir
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Importar r2_score

    # Método para entrenar y predecir
    def entrenar_y_predecir(self, df, feature_columns, target_columns, dias):
        X = df[feature_columns].values # Características
        y = df[target_columns].values # Objetivo

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # División de datos en entrenamiento y prueba
        scaler = StandardScaler() # Escalador estándar
        X_train = scaler.fit_transform(X_train) # Escalar los datos de entrenamiento
        X_test = scaler.transform(X_test) # Escalar los datos de prueba

        self.model.fit(X_train, y_train) # Entrenar el modelo
        y_pred_test = self.model.predict(X_test) # Predecir en los datos de prueba

        mae_test = mean_absolute_error(y_test, y_pred_test) # Error absoluto medio
        mse_test = mean_squared_error(y_test, y_pred_test) # Error cuadrático medio
        r2_test = r2_score(y_test, y_pred_test) # R²

        y_pred_test_mapped = pd.DataFrame(y_pred_test, columns=target_columns) # Crear un DataFrame con las predicciones
        y_pred_test_mapped['weather'] = y_pred_test_mapped['weather'].apply(
            lambda x: self.weather_mapping.get(max(1, min(5, int(x))), 'Unknown') # Mapear los valores de clima
        )
        y_pred_test_mapped['cloudiness'] = y_pred_test_mapped['cloudiness'].apply(
            lambda x: self.cloudiness_mapping.get(max(1, min(3, int(x))), 'Unknown') # Mapear los valores de nubosidad
        )
        
        y_pred_test_mapped_limited = y_pred_test_mapped.head(dias) # Limitar los resultados a los días seleccionados
        return y_pred_test_mapped, y_pred_test_mapped_limited, mae_test, mse_test, r2_test
