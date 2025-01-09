import pandas as pd
import sqlite3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE  # Importar SMOTE para balancear las clases

class GradientBoosterClassifier:
    def __init__(self):
        # Inicializar el modelo Gradient Boosting
        self.model = GradientBoostingClassifier(
            random_state=42, n_estimators=100, max_depth=10, learning_rate=0.1
        )
        self.label_encoder = LabelEncoder()  # Codificador para las clases categóricas
    
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
        df.drop(columns=['date_id', 'weather_id', 'cloudiness_id', 'estacion_id'], inplace=True)
        cols = ['date'] + [col for col in df.columns if col != 'date']
        df = df[cols]
        connection.close()  # Cerrar la conexión
        return df
    
    def procesar_datos(self, df):
        # Crear una copia del DataFrame para evitar modificar el original
        df = df.copy()
        # Codificar variables categóricas
        df['weather_encoded'] = self.label_encoder.fit_transform(df['weather'])
        df['cloudiness'] = pd.Categorical(df['cloudiness']).codes
        # Crear una nueva característica: rango de temperatura
        df['temp_range'] = df['temp_max'] - df['temp_min']
        return df

    def entrenar_y_predecir(self, df, feature_columns, target_column):
        # Seleccionar características y objetivo
        X = df[feature_columns].values
        y = df['weather_encoded']  # Columna codificada como objetivo

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Aplicar SMOTE para balancear las clases
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Escalar las características
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Entrenar el modelo
        self.model.fit(X_train, y_train)

        # Predicciones sobre el conjunto de prueba
        y_pred_test = self.model.predict(X_test)

        # Generar reporte de clasificación
        weather_classes = self.label_encoder.classes_  # Obtener las clases originales
        report = classification_report(y_test, y_pred_test, target_names=weather_classes)

        # Decodificar las predicciones para interpretación
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred_test)

        # Calcular la precisión del modelo
        accuracy = accuracy_score(y_test, y_pred_test)

        return y_test_decoded, y_pred_decoded, accuracy, report
