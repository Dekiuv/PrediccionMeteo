import plotly.graph_objects as go # Librería para gráficos
import streamlit as st # Librería para la interfaz

def grafica(y_pred_limited, dias):
    # Representación gráfica de los días seleccionados
    st.subheader("Predicción Gráfica")

    # Crear la figura
    fig = go.Figure()

    # Grafico lineal para temperaturas
    fig.add_trace(go.Scatter(
        x=y_pred_limited.index, 
        y=y_pred_limited['temp_min'], 
        mode='lines', 
        name='Temp Mín', 
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=y_pred_limited.index, 
        y=y_pred_limited['temp_max'], 
        mode='lines', 
        name='Temp Máx', 
        line=dict(color='red')
    ))

    # Grafico de barras para precipitación
    fig.add_trace(go.Bar(
        x=y_pred_limited.index, 
        y=y_pred_limited['precipitation'], 
        name='Precipitación',
        marker=dict(color='skyblue')
    ))

    # Personalización de la gráfica
    fig.update_layout(
        title="Predicción de Temperatura y Precipitación",
        xaxis_title="Días",
        yaxis_title="Valores: °C / mm",
        xaxis=dict(
            tickvals=y_pred_limited.index, 
            ticktext=[f'Dia{i+1}' for i in range(dias)]
        ),
        barmode='group',  # Las barras serán mostradas en grupo con las líneas
    )
    # Mostrar la gráfica
    st.plotly_chart(fig)