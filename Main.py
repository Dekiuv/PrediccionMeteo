import streamlit as st  # Libreria para la creación de la interfaz
import Entrenamiento as ent  # Libreria para la carga de datos y entrenamiento del modelo
import Grafica as graf  # Libreria para la creación de la gráfica
import GradientBooster as gbc  # Importar la clase GradientBooster
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Simulador Meteorológico",
    page_icon="🌤️",
    layout="wide"
)

def main():
    st.title("Simulador Meteorológico - Grupo 3")
    st.write("Aplicación para realizar predicciones meteorológicas.")

    # Ruta de la base de datos
    db_path = "CSV/Prediccion.db"

    # Selección del modelo
    modelo_seleccionado = st.selectbox("Selecciona el modelo de predicción", ["Random Forest", "Gradient Boosting (Clasificación)"])

    if modelo_seleccionado == "Random Forest":
        simulador = ent.SimuladorMeteorologico(db_path)  # Clase de Random Forest
    elif modelo_seleccionado == "Gradient Boosting (Clasificación)":
        simulador = gbc.GradientBooster()  # Clase de Gradient Boosting para clasificación

    # Cargar los datos de la base de datos
    df = simulador.cargar_datos()  # Cargar los datos de la base de datos

    # Selección del usuario de la estación del año
    estaciones = ['General', 'Otoño', 'Invierno', 'Primavera', 'Verano']
    estacion_seleccionada = st.selectbox("Selecciona la estación del año", estaciones)

    # Solicitar al usuario el número de días que desea predecir
    dias = st.number_input("Número de días a predecir (Mínimo 1 día, Máximo 30 días)", min_value=1, value=7, max_value=30)

    # Filtrar los datos según la estación seleccionada
    if estacion_seleccionada == 'General':
        df_estacion = df
    else:
        df_estacion = df[df['estacion'] == estacion_seleccionada]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Datos cargados para la estación: {estacion_seleccionada}")
        if not df_estacion.empty:
            st.dataframe(df_estacion)
        else:
            st.write("Selecciona una estación para ver los datos.")

        if st.button("Realizar predicción"):
            with st.spinner('Realizando predicciones...'):
                df_procesado = simulador.procesar_datos(df_estacion)  # Procesar los datos
                feature_columns = ["temp_min", "humidity", "pressure", "solar_radiation", "precipitation", "cloudiness", "temp_range"]

                if modelo_seleccionado == "Random Forest":
                    target_columns = ["temp_min", "temp_max", "precipitation", "weather", "cloudiness"]
                    y_pred, y_pred_limited, mae, mse, r2 = simulador.entrenar_y_predecir(df_procesado, feature_columns, target_columns, dias)

                    # Mostrar resultados de Random Forest
                    with st.expander("Información predicción"):
                        st.subheader("Resultados de la predicción")
                        st.write(f"MAE en prueba: {mae:.2f}")
                        st.write(f"MSE en prueba: {mse:.2f}")
                        st.write(f"R² en prueba: {r2:.2f}")

                    with col2:
                        graf.grafica(y_pred_limited, dias)  # Mostrar la gráfica
                        with st.expander("Predicciones"):
                            st.subheader("Predicciones")
                            for index, row in y_pred_limited.iterrows():
                                st.write(f"Día {index + 1} | Temp Mín: {row['temp_min']:.2f}°C | "
                                         f"Temp Máx: {row['temp_max']:.2f}°C | "
                                         f"Precipitación: {row['precipitation']:.2f} mm | "
                                         f"Condición: {row['weather']} | Nubosidad: {row['cloudiness']}")

                            csv = y_pred.to_csv(index=False)
                            st.download_button(
                                label="Descargar todas las predicciones como CSV",
                                data=csv,
                                file_name=f"Predicciones_{estacion_seleccionada}.csv",
                                mime="text/csv"
                            )

                elif modelo_seleccionado == "Gradient Boosting (Clasificación)":
                    y_test, y_pred, accuracy, report = simulador.entrenar_y_predecir(df_procesado, feature_columns, "weather")

                    # Mostrar resultados de Gradient Boosting
                    st.subheader("Resultados de la predicción")
                    st.write(f"Precisión del modelo: {accuracy:.2f}")
                    st.text("Reporte de clasificación:")
                    st.text(report)

                    # Mostrar predicciones reales vs predichas
                    predicciones_df = pd.DataFrame({
                        "Real": y_test,
                        "Predicción": y_pred
                    })
                    st.dataframe(predicciones_df)

# Llamada a la función principal
if __name__ == "__main__":
    main()
