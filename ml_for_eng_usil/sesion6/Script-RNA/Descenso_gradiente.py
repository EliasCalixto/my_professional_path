# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:18:39 2025

@author: IVAN
"""

"""
Visualizar cómo funciona el descenso de gradiente para minimizar
 una función cuadrática.
"""

import numpy as np
import matplotlib.pyplot as plt


# Función cuadrática simple: f(x) = (x - 3)^2
def funcion(x):
    return (x - 3)**2

# Derivada de la función
def derivada(x):
    return 2 * (x - 3)

# Descenso de gradiente
x = 0.0  # Valor inicial
lr = 0.1  # Tasa de aprendizaje (learning rate)
historial = [x]

for _ in range(20):
    x = x - lr * derivada(x)
    historial.append(x)

# Visualización
x_vals = np.linspace(-1, 6, 100)
y_vals = funcion(x_vals)

plt.plot(x_vals, y_vals, label='f(x) = (x - 3)^2')
plt.plot(historial, [funcion(h) for h in historial], 'ro--', label='Descenso de gradiente')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Optimización con Descenso de Gradiente")
plt.legend()
plt.show()
