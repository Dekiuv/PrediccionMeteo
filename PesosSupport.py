# Calcular los pesos inversamente proporcionales al soporte de las clases
# El soporte indica el número de muestras disponibles para cada clase.
support = {
    "Tormenta": 1817,
    "Lluvia": 1894,
    "Nublado": 1179,
    "Niebla": 88,
    "Soleado": 22
}

# Calcular el número total de muestras sumando los valores del soporte
total_samples = sum(support.values())

# Calcular los pesos inversamente proporcionales al soporte
# Cuanto menor sea el soporte de una clase, mayor será su peso para balancear la importancia de esa clase.
weights = {key: total_samples / support[key] for key in support}

# Mostrar los pesos
# Los pesos representan la importancia relativa de cada clase, siendo mayores para clases con menos muestras.
print(weights)