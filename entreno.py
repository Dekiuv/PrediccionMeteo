# entreno.py

import os
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Función para cargar datos desde SQLite
def cargar_datos():
    db_path = "CSV/Prediccion.db"  # Cambiar a la ruta correcta
    connection = sqlite3.connect(db_path)
    query_valores = "SELECT * FROM ValoresLimpios"
    df_valores = pd.read_sql_query(query_valores, connection)
    connection.close()
    return df_valores

# Función para preparar los datos
def preparar_datos(df, feature_columns):
    target_column = 'weather_id'  # Columna que se va a predecir

    X = df[feature_columns]  # Características
    y = df[target_column]  # Etiquetas

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Aplicar SMOTE para balancear las clases
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)  # Escalar los datos de entrenamiento
    X_test_scaled = scaler.transform(X_test)  # Escalar los datos de prueba

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler

# Pesos de clase definidos
class_weights = {
    1: 2.7517886626307098,  # Tormenta
    2: 2.6399155227032733,  # Lluvia
    3: 4.2408821034775235,  # Nublado
    4: 56.81818181818182,   # Niebla
    5: 227.27272727272728   # Soleado
}

# Función para optimizar hiperparámetros y entrenar el modelo
def optimizar_y_entrenar(X_train, y_train):
    param_distributions = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': [class_weights]  # Usar los pesos de clase
    }
    # Optimizar modelo con RandomizedSearchCV
    model = SVC(probability=True)  # Activar las probabilidades
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

# Función para guardar el modelo entrenado
def guardar_modelo(modelo, nombre_archivo):
    with open(nombre_archivo, 'wb') as f:
        joblib.dump(modelo, f)
    print(f"Modelo guardado en {nombre_archivo}")

# Función principal para ejecutar el entrenamiento
def main():
    # Cargar datos
    df_valores = cargar_datos()

    # Características fijas para el modelo
    features_options = ['precipitation', 'wind', 'humidity', 'visibility']
    selected_features = features_options  # Usamos estas características siempre

    # Preparar los datos con las características seleccionadas
    X_train, X_test, y_train, y_test, scaler = preparar_datos(df_valores, selected_features)

    # Entrenar el modelo
    best_model = optimizar_y_entrenar(X_train, y_train)

    # Guardar el modelo entrenado
    modelo_guardado = "ModeloEntrenado.pkl"
    guardar_modelo(best_model, modelo_guardado)

if __name__ == "__main__":
    main()
