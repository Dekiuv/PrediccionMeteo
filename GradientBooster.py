import pandas as pd
import sqlite3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class GradientBooster:
    def __init__(self):
        # Inicializar el modelo Gradient Boosting para clasificación
        self.model = GradientBoostingClassifier(
            random_state=42, n_estimators=100, max_depth=10, learning_rate=0.1
        )
        # Inicializar el codificador de etiquetas para variables categóricas
        self.label_encoder = LabelEncoder()
        
    def cargar_datos(self):
        # Conectar a la base de datos SQLite
        connection = sqlite3.connect("CSV/Prediccion.db")
        query = """
        SELECT vl.*, d.date, s.estacion, c.cloudiness AS cloudiness, w.weather
        FROM ValoresLimpios vl
        JOIN dates d ON vl.date_id = d.date_id
        JOIN seasons s ON vl.estacion_id = s.estacion_id
        JOIN cloudiness c ON vl.cloudiness_id = c.cloudiness_id
        JOIN weather w ON vl.weather_id = w.weather_id
        """
        # Cargar los datos de la base de datos
        df = pd.read_sql(query, connection)
        # Eliminar columnas no necesarias
        df.drop(columns=['date_id', 'weather_id', 'cloudiness_id', 'estacion_id'], inplace=True)
        # Reordenar columnas para una mejor presentación
        cols = ['date'] + [col for col in df.columns if col != 'date']
        df = df[cols]
        connection.close()  # Cerrar la conexión a la base de datos
        return df
    
    def procesar_datos(self, df):
        # Trabajar con una copia del DataFrame para evitar modificar el original
        df = df.copy()
        # Codificar la columna 'weather' en valores numéricos
        df['weather_encoded'] = self.label_encoder.fit_transform(df['weather'])
        # Codificar la columna 'cloudiness' en valores numéricos
        df['cloudiness'] = pd.Categorical(df['cloudiness']).codes
        # Calcular el rango de temperatura (temp_max - temp_min)
        df['temp_range'] = df['temp_max'] - df['temp_min']
        return df

    def entrenar_y_predecir(self, df, feature_columns, target_column):
        # Seleccionar las características (X) y el objetivo codificado (y)
        X = df[feature_columns].values
        y = df['weather_encoded']  # Usar la columna codificada para el clima

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar las características para normalizar los valores
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Entrenar el modelo
        self.model.fit(X_train, y_train)
        # Realizar predicciones sobre los datos de prueba
        y_pred_test = self.model.predict(X_test)

        # Obtener las clases originales de la columna 'weather'
        weather_classes = df['weather'].unique().tolist()
        
        # Generar el reporte de clasificación
        try:
            report = classification_report(y_test, y_pred_test, target_names=weather_classes)
        except Exception as e:
            print(f"Error al generar el reporte de clasificación: {e}")
            report = classification_report(y_test, y_pred_test)

        # Decodificar las etiquetas predichas y reales a los nombres originales
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred_test)

        # Calcular la precisión del modelo
        accuracy = accuracy_score(y_test, y_pred_test)

        return y_test_decoded, y_pred_decoded, accuracy, report
