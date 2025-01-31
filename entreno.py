# Importamos todas las librerías necesarias
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import time

# Función para cargar datos desde SQLite
def cargar_datos():
    db_path = "CSV/Prediccion.db"
    connection = sqlite3.connect(db_path)
    query_valores = "SELECT * FROM ValoresLimpios"
    df_valores = pd.read_sql_query(query_valores, connection)
    connection.close()
    return df_valores

# Función para preparar los datos
def preparar_datos(df):

    # Seleccionar las características (X) y la variable objetivo (y)
    X = df['precipitation', 'wind', 'humidity', 'visibility']  # Variables independientes (características seleccionadas)
    y = df['weather_id']  # Variable a predecir (columna objetivo)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Escalar los datos para que tengan una media de 0 y una desviación estándar de 1
    # El escalador solo se ajustara con los datos de entrenamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)  # Escalar los datos de entrenamiento
    X_test_scaled = scaler.transform(X_test)  # Escalar los datos de prueba

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler

# Pesos definidos para balancear las clases desbalanceadas en el modelo, procedimiento en PesosSupport.py
class_weights = {
    1: 2.7517886626307098,  # Tormenta
    2: 2.6399155227032733,  # Lluvia
    3: 4.2408821034775235,  # Nublado
    4: 56.81818181818182,   # Niebla
    5: 227.27272727272728   # Soleado
}

# Función para optimizar hiperparámetros y entrenar el modelo
def optimizar_y_entrenar(X_train, y_train):
    # Definir algunos hiperparametros para el modelo
    param_distributions = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': [class_weights]  # Usar los pesos definidos
    }

    # Crear una instancia del modelo SVM
    model = SVC(probability=True)  # Habilitar probabilidad para predicciones probabilísticas
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    
    # Registrar el tiempo de entreno
    start_time = time.time()
    random_search.fit(X_train, y_train) # Ajustar el modelo a los datos
    end_time = time.time()

    # Calcular el tiempo total de entreno
    elapsed_time = end_time - start_time
    print(f"Tiempo de entrenamiento: {elapsed_time:.2f} segundos")

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

    # Definir las características que utilizaremos
    features_options = ['precipitation', 'wind', 'humidity', 'visibility'] # Características seleccionadas

    # Preparar los datos con las características seleccionadas
    X_train, y_train = preparar_datos(df_valores, features_options)

    # Entrenar el modelo y buscar los mejores hiperparámetros
    best_model = optimizar_y_entrenar(X_train, y_train)

    # Guardar el modelo entrenado
    modelo_guardado = "ModeloEntrenado.pkl"
    guardar_modelo(best_model, modelo_guardado)

if __name__ == "__main__":
    main()