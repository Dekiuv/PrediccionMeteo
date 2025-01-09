import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

# Función para cargar datos desde SQLite
@st.cache_data
def cargar_datos():
    db_path = "CSV/Prediccion.db"  # Cambiar a la ruta correcta
    connection = sqlite3.connect(db_path)
    query_valores = "SELECT * FROM ValoresLimpios"
    query_weather = "SELECT * FROM weather"
    
    df_valores = pd.read_sql_query(query_valores, connection)
    df_weather = pd.read_sql_query(query_weather, connection)
    
    connection.close()
    return df_valores, df_weather

# Función para entrenar el modelo y realizar predicciones
def entrenar_y_predecir(df, tolerance=0.7):
    feature_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']  # Características
    target_column = 'weather_id'

    X = df[feature_columns]
    y = df[target_column]

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar el modelo
    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train_resampled)

    # Predicciones
    y_pred = model.predict(X_test_scaled)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calcular accuracy basado en un margen de tolerancia
    correct_predictions = sum(abs(y_test - y_pred) <= tolerance)
    total_predictions = len(y_test)
    accuracy = (correct_predictions / total_predictions) * 100  # Accuracy en %

    # Crear un DataFrame con resultados
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return mae, mse, r2, accuracy, results

# Configuración de la aplicación Streamlit
st.title("Simulador Meteorológico")
st.write("Esta aplicación permite visualizar las predicciones del modelo meteorológico.")

# Cargar los datos
df_valores, df_weather = cargar_datos()

# Crear un diccionario para mapear weather_id a descripciones
weather_mapping = dict(zip(df_weather['weather_id'], df_weather['weather']))

st.write("Datos cargados correctamente:")
st.dataframe(df_valores.head(10))  # Muestra las primeras 10 filas de los datos

# Seleccionar el número de días de predicción
dias = st.slider("Seleccione el número de días para visualizar las predicciones:", min_value=1, max_value=50, value=10)

# Entrenar y predecir
if st.button("Entrenar modelo y predecir"):
    st.write("Entrenando el modelo y generando predicciones...")
    mae, mse, r2, accuracy, results = entrenar_y_predecir(df_valores)

    # Mapear weather_id a descripciones en los resultados
    results['Actual'] = results['Actual'].map(weather_mapping)
    results['Predicted'] = results['Predicted'].round().astype(int).map(weather_mapping)

    # Mostrar métricas
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.2f}")
    st.write(f"**Accuracy:** {accuracy:.2f}%")

    # Mostrar resultados limitados por los días seleccionados
    st.write("**Resultados de las predicciones:**")
    st.dataframe(results.head(dias))  # Muestra las primeras N predicciones según lo seleccionado