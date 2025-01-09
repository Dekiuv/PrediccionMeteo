import streamlit as st  # Librería para la creación de la interfaz
import Entrenamiento as ent  # Librería para la carga de datos y entrenamiento del modelo
import Grafica as graf  # Librería para la creación de la gráfica
import GradientBooster as gbc  # Importar la clase GradientBooster
import pandas as pd
from sklearn.metrics import classification_report

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
    modelo_seleccionado = st.selectbox("Selecciona el modelo de predicción", ["Random Forest", "Gradient Boosting"])

    if modelo_seleccionado == "Random Forest":
        simulador = ent.SimuladorMeteorologico(db_path)  # Clase de Random Forest
    elif modelo_seleccionado == "Gradient Boosting":
        simulador = gbc.GradientBoosterClassifier()  # Clase de Gradient Boosting con SMOTE

    # Cargar los datos de la base de datos
    df = simulador.cargar_datos()  # Cargar los datos de la base de datos

    # Selección del usuario de la estación del año
    estaciones = ['General', 'Otoño', 'Invierno', 'Primavera', 'Verano']
    estacion_seleccionada = st.selectbox("Selecciona la estación del año", estaciones)

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

                feature_columns = [
                    "temp_max", "temp_min", "precipitation", "wind", "humidity",
                    "pressure", "solar_radiation", "visibility", "cloudiness",
                    "temp_range", "humidity_temp_ratio", "pressure_change", "wind_visibility_ratio"
                ]

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

                elif modelo_seleccionado == "Gradient Boosting":
                    y_test, y_pred, accuracy, report = simulador.entrenar_y_predecir(df_procesado, feature_columns, "weather")

                    with col2:
                        # Mostrar resultados de Gradient Boosting
                        st.markdown("## Resultados de la Predicción")
                        st.markdown(f"### Precisión del Modelo: **{accuracy:.2f}**")
                        st.markdown("### Reporte de Clasificación")

                        def reporte_a_dataframe(report):
                            report_dict = classification_report(y_test, y_pred, output_dict=True)
                            df_report = pd.DataFrame(report_dict).transpose()
                            return df_report

                        df_report = reporte_a_dataframe(report)
                        st.dataframe(df_report.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

                        st.markdown("---")

                        st.markdown("### Métricas Clave")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label="Precisión Promedio", value=f"{df_report.loc['macro avg', 'precision']:.2f}")
                        col2.metric(label="Recall Promedio", value=f"{df_report.loc['macro avg', 'recall']:.2f}")
                        col3.metric(label="F1-Score Promedio", value=f"{df_report.loc['macro avg', 'f1-score']:.2f}")

                        st.markdown("---")

                        st.markdown("### Predicciones Reales vs Predichas")
                        df_predicciones = pd.DataFrame({
                            "Real": y_test,
                            "Predicción": y_pred
                        })

                        def resaltar_predicciones(val):
                            return ['background-color: #31c852' if val['Real'] == val['Predicción'] else 'background-color: #e55361'] * len(val)

                        st.dataframe(df_predicciones.style.apply(resaltar_predicciones, axis=1))

# Llamada a la función principal
if __name__ == "__main__":
    main()
