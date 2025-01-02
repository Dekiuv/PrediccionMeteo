import streamlit as st #Libreria para la creación de la interfaz
import Entrenamiento as ent #Libreria para la carga de datos y entrenamiento del modelo
import Grafica as graf #Libreria para la creación de la gráfica

# Configuración cabezera de la página
st.set_page_config(
    page_title="Simulador Meteorológico",  # Título de la página en el navegador
    page_icon="🌤️",  # Icono de la página
    layout="wide"  # Diseño de la página (en este caso, "ancho")
)

# Función principal
def main():
    st.title("Simulador Meteorológico - Grupo 3") # Título de la página
    st.write("Aplicación para realizar predicciones meteorológicas utilizando un modelo entrenado RandomForest.") # Descripción de la aplicación

    # Cargar los datos de la base de datos
    db_path = "CSV/Prediccion.db" # Ruta de la base de datos
    simulador = ent.SimuladorMeteorologico(db_path) # Instancia de la clase SimuladorMeteorologico
    df = simulador.cargar_datos() # Cargar los datos de la base de datos

    # Selección del usuario de la estación del año
    estaciones = ['General', 'Otoño', 'Invierno', 'Primavera', 'Verano'] # Opciones de estaciones disponibles
    estacion_seleccionada = st.selectbox("Selecciona la estación del año", estaciones) # Desplegable para seleccionar la estación

    # Solicitar al usuario el número de días que desea predecir
    dias = st.number_input("Número de días a predecir (Minimo 1 dia, Maximo 30 dias)", min_value=1, value=7, max_value=30)

    # Filtrar los datos según la estación seleccionada
    if estacion_seleccionada == 'General':
        df_estacion = df 
    elif estacion_seleccionada == 'Otoño':
        df_estacion = df[df['estacion'] == 'Otoño']
    elif estacion_seleccionada == 'Invierno':
        df_estacion = df[df['estacion'] == 'Invierno']
    elif estacion_seleccionada == 'Primavera':
        df_estacion = df[df['estacion'] == 'Primavera']
    elif estacion_seleccionada == 'Verano':
        df_estacion = df[df['estacion'] == 'Verano']

    col1, col2 = st.columns(2) # Dividir la página en dos columnas

    with col1: # Columna 1 (parte izquierda)
        st.subheader(f"Datos cargados para la estación: {estacion_seleccionada}") 
        if estacion_seleccionada != ' ': 
            st.dataframe(df_estacion) 
        else:
            st.write("Selecciona una estación para ver los datos")

        if st.button("Realizar predicción"):
            with st.spinner('Realizando predicciones...'):
                    df_procesado = simulador.procesar_datos(df_estacion) # Procesar los datos de la estación seleccionada
                    feature_columns = ["temp_min", "humidity", "pressure", "solar_radiation", "precipitation", "cloudiness", "temp_range"] # Columnas de características
                    target_columns = ["temp_min", "temp_max", "precipitation", "weather", "cloudiness"] # Columnas objetivo a predecir
                    y_pred, y_pred_limited, mae, mse, r2 = simulador.entrenar_y_predecir(df_procesado, feature_columns, target_columns, dias) # Entrenar y predecir

            # Desplegable para mostrar la Información de la predicción
            with st.expander("Información predicción"):
                st.subheader("Resultados de la predicción")
                st.write(f"MAE en prueba: {mae:.2f}") # Mostrar el error absoluto medio
                st.write(f"MSE en prueba: {mse:.2f}") # Mostrar el error cuadrático medio
                st.write(f"R² en prueba: {r2:.2f}")  # Mostrar el R²

            # Columna 2 (parte derecha)
            with col2:
                graf.grafica(y_pred_limited, dias) # Mostrar la gráfica de la predicción
                with st.expander("Predicciones"):
                    # Mostrar las predicciones con temperatura mínima, máxima, precipitación y condiciones
                    st.subheader("Predicciones con temperatura mínima, máxima, precipitación, condiciones y nubosidad")
                    for index, row in y_pred_limited.iterrows():
                        # Mostrar los datos de cada día
                        st.write(f"Dia{index+1} | Temp Mín: {row['temp_min']:.2f}°C | Temp Máx: {row['temp_max']:.2f}°C | Precipitación: {row['precipitation']:.2f}mm | Condición: {row['weather']} | Nubosidad: {row['cloudiness']}")

                    csv = y_pred.to_csv(index=False)  # Guardamos todas las predicciones (no limitadas)
                    st.download_button(
                        label="Descargar todas las predicciones como CSV",
                        data=csv,
                        file_name=f"Predicciones_{estacion_seleccionada}.csv",
                        mime="text/csv"
                    )

# Llamada a la función principal
if __name__ == "__main__":
    main()