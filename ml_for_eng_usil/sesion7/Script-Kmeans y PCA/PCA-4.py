# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 11:24:40 2025

@author: IVAN
"""

# Encontrar el número óptimo de Componentes Principales con el 95% de varianza (información relevante)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Cargar y escalar datos
data = load_breast_cancer()
X = data.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA (todos los componentes)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Varianza explicada
# Cuanta información (%) captura cada componente principal y acumulada
varianza = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza) # con 9 componentes ya se alcanza el 95%

# Determinar número de componentes para el 95% de varianza
# 95% es para un modelado sin mucha pérdida de información relevante (a veces 99% ó 90%)
n_componentes_95 = np.argmax(varianza_acumulada >= 0.95) + 1

# Graficar
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% de varianza')
plt.axvline(x=n_componentes_95, color='g', linestyle='--', label=f'{n_componentes_95} componentes')
plt.title('Varianza explicada acumulada - PCA')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("# de componentes principales (varianza 95%)", n_componentes_95)
