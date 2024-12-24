import streamlit as st
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

    def cargar_datos(self):
        """
        Carga los datos de la tabla ValoresLimpios desde la base de datos.
        """
        query = """
        SELECT vl.*, d.date, s.estacion, c.cloudiness AS cloudiness, w.weather
        FROM ValoresLimpios vl
        JOIN dates d ON vl.date_id = d.date_id
        JOIN seasons s ON vl.estacion_id = s.estacion_id
        JOIN cloudiness c ON vl.cloudiness_id = c.cloudiness_id
        JOIN weather w ON vl.weather_id = w.weather_id
        """
        df = pd.read_sql(query, self.connection)
        
        # Eliminamos las columnas de los IDs, ya que ahora tenemos los valores correspondientes
        df.drop(columns=['date_id', 'weather_id', 'cloudiness_id', 'estacion_id'], inplace=True)
        
        # Reordenar las columnas para que 'date' esté primero
        cols = ['date'] + [col for col in df.columns if col != 'date']
        df = df[cols]
        
        return df

    def procesar_datos(self, df):
        """
        Preprocesa los datos de la tabla ValoresLimpios.
        """
        # Filtrar valores nulos
        df.dropna(subset=["temp_max", "temp_min", "precipitation", "weather", "cloudiness"], inplace=True)

        # Codificar las columnas categóricas
        le_weather = LabelEncoder()
        df['weather'] = le_weather.fit_transform(df['weather'])

        le_nubosidad = LabelEncoder()
        df['cloudiness'] = le_nubosidad.fit_transform(df['cloudiness'])

        # Calcular rango de temperatura
        df['temp_range'] = df['temp_max'] - df['temp_min']

        return df

    def entrenar_y_predecir(self, df, feature_columns, target_columns):
        """
        Entrena el modelo de predicción y realiza predicciones.
        """
        X = df[feature_columns].values
        y = df[target_columns].values  # Ahora 'y' es un DataFrame con múltiples columnas de objetivos

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizar características
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Entrenar el modelo de RandomForest
        self.model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred_test = self.model.predict(X_test)

        # Evaluar el rendimiento
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Mapear las predicciones
        y_pred_test_mapped = pd.DataFrame(y_pred_test, columns=target_columns)
        y_pred_test_mapped['weather'] = y_pred_test_mapped['weather'].apply(
            lambda x: self.weather_mapping.get(max(1, min(5, int(x))), 'Unknown')
        )

        return y_pred_test_mapped, mae_test, mse_test

# Interfaz de Streamlit
def main():
    st.title("Simulador Meteorológico")
    st.write("Aplicación para realizar predicciones meteorológicas utilizando un modelo entrenado.")

    db_path = "CSV/Prediccion.db"  # Ruta a tu base de datos SQLite
    simulador = SimuladorMeteorologico(db_path)

    # Cargar la tabla ValoresLimpios
    df = simulador.cargar_datos()

    # Mostrar la tabla a los usuarios
    st.subheader("Datos de la tabla 'ValoresLimpios'")
    st.dataframe(df)

    # Botón para realizar predicciones
    if st.button("Realizar predicción"):
        # Preprocesar los datos
        df_procesado = simulador.procesar_datos(df)

        # Definir las columnas de características y objetivos
        feature_columns = ["temp_min", "humidity", "pressure", "solar_radiation", "precipitation", "cloudiness", "temp_range"]
        target_columns = ["temp_min", "temp_max", "precipitation", "weather", "cloudiness"]

        # Realizar la predicción
        y_pred, mae, mse = simulador.entrenar_y_predecir(df_procesado, feature_columns, target_columns)

        # Mostrar los resultados
        st.subheader("Resultados de la predicción")
        st.write(f"MAE en prueba: {mae}")
        st.write(f"MSE en prueba: {mse}")
        st.write("Predicciones:")
        st.dataframe(y_pred)

if __name__ == "__main__":
    main()