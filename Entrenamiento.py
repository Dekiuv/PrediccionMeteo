import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SimuladorMeteorologico:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=3)

        self.weather_mapping = {
            1: "storm",
            2: "rain",
            3: "cloudy",
            4: "fog",
            5: "sun"
        }

        self.cloudiness_mapping = {
            1: "parcialmente nublado",
            2: "cubierto",
            3: "despejado"
        }

    def cargar_datos(self):
        query = """
        SELECT vl.*, d.date, s.estacion, c.cloudiness AS cloudiness, w.weather
        FROM ValoresLimpios vl
        JOIN dates d ON vl.date_id = d.date_id
        JOIN seasons s ON vl.estacion_id = s.estacion_id
        JOIN cloudiness c ON vl.cloudiness_id = c.cloudiness_id
        JOIN weather w ON vl.weather_id = w.weather_id
        """
        df = pd.read_sql(query, self.connection)
        df.drop(columns=['date_id', 'weather_id', 'cloudiness_id', 'estacion_id'], inplace=True)
        cols = ['date'] + [col for col in df.columns if col != 'date']
        df = df[cols]
        return df

    def procesar_datos(self, df):
        df.dropna(subset=["temp_max", "temp_min", "precipitation", "weather", "cloudiness"], inplace=True)
        le_weather = LabelEncoder()
        df['weather'] = le_weather.fit_transform(df['weather'])
        le_nubosidad = LabelEncoder()
        df['cloudiness'] = le_nubosidad.fit_transform(df['cloudiness'])
        df['temp_range'] = df['temp_max'] - df['temp_min']
        return df
    
    def entrenar_y_predecir(self, df, feature_columns, target_columns, dias):
        X = df[feature_columns].values
        y = df[target_columns].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.model.fit(X_train, y_train)
        y_pred_test = self.model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        y_pred_test_mapped = pd.DataFrame(y_pred_test, columns=target_columns)
        y_pred_test_mapped['weather'] = y_pred_test_mapped['weather'].apply(
            lambda x: self.weather_mapping.get(max(1, min(5, int(x))), 'Unknown')
        )
        y_pred_test_mapped['cloudiness'] = y_pred_test_mapped['cloudiness'].apply(
            lambda x: self.cloudiness_mapping.get(max(1, min(3, int(x))), 'Unknown')
        )
        # Limitar los resultados a los días seleccionados para la visualización
        y_pred_test_mapped_limited = y_pred_test_mapped.head(dias)  # Aquí limitamos la cantidad de días a los indicados
        return y_pred_test_mapped, y_pred_test_mapped_limited, mae_test, mse_test