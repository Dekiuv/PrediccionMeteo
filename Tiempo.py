import numpy as np

# Tiempos de las 3 validaciones cruzadas (en segundos)
tiempos = [10464, 10572, 12186]

# Calcular el tiempo promedio
tiempo_promedio = np.mean(tiempos)

# Imprimir los mejores parámetros
print("Mejores parámetros Hiperparametros:")
print(f"- C: 100")
print(f"- Gamma: 0.1")
print(f"- Kernel: rbf")
print(f"- Class_weight: Diccionario personalizado")
print(f"- Tiempo promedio: {tiempo_promedio:.2f} segundos")
