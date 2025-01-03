# Simulador de Predicción Meteorológica con Big Data 🌤️

## Descripción 📝
Este proyecto tiene como objetivo desarrollar un simulador de predicción meteorológica que aprovecha el poder de Big Data y aprendizaje automático para predecir condiciones climáticas a corto plazo con alta precisión. Utilizamos:  
- **Random Forest Regressor**🌲 de Scikit-learn para entrenar un modelo de predicción de temperatura y condiciones climáticas basado en características como temperatura máxima, mínima y nubosidad.
- **Label Encoding** 🔢 para transformar las etiquetas de clima y nubosidad en valores numéricos para su uso en el modelo.
- **StandardScaler** ⚖️ para normalizar las características y mejorar la precisión del modelo de predicción.
- **Métricas de rendimiento** 📈 como el Error Absoluto Medio (MAE), Error Cuadrático Medio (MSE) y R² para evaluar la efectividad del modelo.

El enfoque principal de este simulador es predecir las condiciones meteorológicas futuras de manera precisa, lo que puede ser útil para diversas aplicaciones como la planificación de eventos, análisis de tendencias climáticas o gestión de recursos en función del clima.

## Tecnologías utilizadas 💻

- Python: Lenguaje de programación principal.
- Sqlite3: Base de datos ligera y autónoma para almacenamiento y manipulación de datos locales.
- Pandas: Manejo y análisis de datos.
- Scikit-learn: Aprendizaje automático para clasificación, regresión y clustering.
- Plotly: Creación de gráficos interactivos y visualizaciones avanzadas en Python.
- Streamlit: Herramienta para crear aplicaciones web interactivas y visualizaciones de datos de manera rápida.

## Versiones necesarias ⚠️
El proyecto utiliza las siguientes librerías y versiones específicas:  
      🐍 **Python** (3.11 o superior)    
      📚 **Pandas** (2.2.3)  
      📚 **Scikit-learn** (1.5.2)  
      📚 **Plotly** (5.24.1)  
      📚 **Streamlit** (1.41.1)  

## Instalación y ejecución 🚀

1. Clona este repositorio en tu máquina local y navega al directorio:

   ```bash
   git clone https://github.com/Dekiuv/PrediccionMeteo.git
   cd PrediccionMeteo
   
2. Una vez clonado el repositorio, ejecuta el siguiente código en el terminal de VSCode:

   ```bash
   python3 -m streamlit run main.py
