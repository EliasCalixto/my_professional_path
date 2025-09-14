
# Método del codo para seleccionar K en el algoritmo de Kmeans

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Cargar datos
data = load_breast_cancer()
X = data.data

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA (opcional, pero se puede usar para visualización)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Método del codo: calcular inercia para distintos valores de k
inercia_valores = []
k_rango = range(1, 11)

for k in k_rango:
    modelo = KMeans(n_clusters=k, random_state=42, init="k-means++")
    modelo.fit(X_scaled)
    inercia_valores.append(modelo.inertia_)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(k_rango, inercia_valores, marker='o')
plt.title("Método del Codo - KMeans")
plt.xlabel("Número de clústeres (k)")
plt.ylabel("Inercia")
plt.grid(True)
plt.tight_layout()
plt.show()
