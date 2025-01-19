# Calcular los pesos inversamente proporcionales al soporte de las clases
support = {
    "Tormenta": 1817,
    "Lluvia": 1894,
    "Nublado": 1179,
    "Niebla": 88,
    "Soleado": 22
}

# Calcular el número total de muestras
total_samples = sum(support.values())

# Calcular pesos inversamente proporcionales al soporte
weights = {key: total_samples / support[key] for key in support}

# Mostrar los pesos
print(weights)
