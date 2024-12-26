import streamlit as st
import plotly.express as px
import Entrenamiento as ent

# Configuración de la página
st.set_page_config(
    page_title="Simulador Meteorológico - G3",  # Título de la página (en la pestaña del navegador)
    page_icon="🌤️",  # Icono de la página (puedes usar emojis o una imagen)
    layout="wide"  # Puedes poner 'centered' para centrar todo el contenido
)

def main():
    st.title("Simulador Meteorológico")
    st.write("Aplicación para realizar predicciones meteorológicas utilizando un modelo entrenado.")

    db_path = "CSV/Prediccion.db"
    simulador = ent.SimuladorMeteorologico(db_path)
    df = simulador.cargar_datos()

    # Selección de la estación
    estaciones = ['General', 'Otoño', 'Invierno', 'Primavera', 'Verano']
    estacion_seleccionada = st.selectbox("Selecciona la estación del año", estaciones)

    # Solicitar al usuario el número de días que desea predecir
    dias = st.number_input("Número de días a predecir (Minimo 1 dia, Maximo 30 dias)", min_value=1, value=7, max_value=30)

    # Filtrar los datos según la estación seleccionada
    if estacion_seleccionada == 'General':
        df_estacion = df  # Sin filtro, mostrar todos los datos
    elif estacion_seleccionada == 'Otoño':
        df_estacion = df[df['estacion'] == 'Otoño']
    elif estacion_seleccionada == 'Invierno':
        df_estacion = df[df['estacion'] == 'Invierno']
    elif estacion_seleccionada == 'Primavera':
        df_estacion = df[df['estacion'] == 'Primavera']
    elif estacion_seleccionada == 'Verano':
        df_estacion = df[df['estacion'] == 'Verano']

    col1, col2 = st.columns(2)
    # Mostrar los datos cargados en la columna izquierda
    with col1:
        st.subheader(f"Datos cargados para la estación: {estacion_seleccionada}")
        st.dataframe(df_estacion)
        if st.button("Realizar predicción"):
            with st.spinner('Cargando las predicciones...'):
                    # Preprocesar los datos de la estación seleccionada
                    df_procesado = simulador.procesar_datos(df_estacion)
                    feature_columns = ["temp_min", "humidity", "pressure", "solar_radiation", "precipitation", "cloudiness", "temp_range"]
                    target_columns = ["temp_min", "temp_max", "precipitation", "weather", "cloudiness"]
                    y_pred, y_pred_limited, mae, mse = simulador.entrenar_y_predecir(df_procesado, feature_columns, target_columns, dias)
            # Desplegable para mostrar la Información de la predicción
            with st.expander("Información predicción"):
                    st.subheader("Resultados de la predicción")
                    st.write(f"MAE en prueba: {mae:.2f}")
                    st.write(f"MSE en prueba: {mse:.2f}")

                # Desplegable para mostrar la Predicción

            with col2:
                    # Representación gráfica de los días seleccionados
                    st.subheader("Predicción Gráfica")
                    fig = px.line(y_pred_limited, x=y_pred_limited.index, y=["temp_min", "temp_max", "precipitation"],
                                title="Predicción de Temperatura y Precipitación",
                                labels={'index': 'Días', 'value': 'Valores: °C / mm'})
                    fig.update_xaxes(tickvals=y_pred_limited.index, ticktext=[f'Dia{i+1}' for i in range(dias)])
                    st.plotly_chart(fig)
                    with st.expander("Predicciones"):
                        # Mostrar las predicciones con temperatura mínima, máxima, precipitación y condiciones
                        st.subheader("Predicciones con temperatura mínima, máxima, precipitación, condiciones y nubosidad")
                        for index, row in y_pred_limited.iterrows():
                            st.write(f"Dia{index+1} | Temp Mín: {row['temp_min']:.2f}°C | Temp Máx: {row['temp_max']:.2f}°C | Precipitación: {row['precipitation']:.2f}mm | Condición: {row['weather']} | Nubosidad: {row['cloudiness']}")
                        # **Guardar todas las predicciones en un archivo CSV**
                        csv = y_pred.to_csv(index=False)  # Guardamos todas las predicciones (no limitadas)
                        st.download_button(
                            label="Descargar todas las predicciones como CSV",
                            data=csv,
                            file_name=f"Predicciones_{estacion_seleccionada}.csv",
                            mime="text/csv"
                            )
if __name__ == "__main__":
    main()
