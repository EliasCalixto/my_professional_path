"""
Created on Sat Apr  5 18:25:11 2025

@author: IVAN


Aplicación de K-Means en el dataset de cáncer de mama,
 visualizar los clústeres  y evaluar los clústers.
 Guarda el modelo, el PCA y el scaler para futuras predicciones (agrupaciones).
"""

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Cargar dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reducir dimensiones con PCA a 2 componentes para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Aplicar K-Means con 2 clústeres
# init = Método de inicialización de centroides (k-means++, random)
kmeans = KMeans(n_clusters=2, random_state=42, init="k-means++")
clusters = kmeans.fit_predict(X_scaled) # sin las etiquetas

# 5. Evaluar clústeres con métricas internas
silhouette = silhouette_score(X_scaled, clusters)
calinski = calinski_harabasz_score(X_scaled, clusters)
davies = davies_bouldin_score(X_scaled, clusters)
inercia = kmeans.inertia_

# 6. Imprimir resultados
print("MÉTRICAS DE EVALUACIÓN DEL CLUSTERING K-Means")
print(f"- Inercia: {inercia:.2f}")
print(f"- Silhouette Score: {silhouette:.4f}")
print(f"- Calinski-Harabasz Score: {calinski:.2f}")
print(f"- Davies-Bouldin Score: {davies:.4f}\n")

# 7. Mostrar número de elementos por clúster
conteo_clusters = np.bincount(clusters)
for i, count in enumerate(conteo_clusters):
    print(f"Clúster {i}: {count} muestras")

# 8. Visualizar clústeres y centroides en PCA 2D
centroides = kmeans.cluster_centers_
centroides_pca = pca.transform(centroides)

plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Accent', alpha=0.6, label='Datos')
# Clúster 0
plt.scatter(X_pca[clusters == 0, 0], X_pca[clusters == 0, 1],
            c='cornflowerblue', label='Clúster 0', alpha=0.6, s=40)
# Clúster 1
plt.scatter(X_pca[clusters == 1, 0], X_pca[clusters == 1, 1],
            c='orange', label='Clúster 1', alpha=0.6, s=40)
# Centroides
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='black', s=100, marker='X', label='Centroides')
plt.title('K-Means clustering con centroides (PCA 2D)')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Guardar modelos
joblib.dump(kmeans, "kmeans_cancer.pkl")
joblib.dump(scaler, "escalador.pkl")
joblib.dump(pca, "pca_2_componentes.pkl")

# 10. Cargar modelos
# scaler = joblib.load("escalador.pkl")
# pca = joblib.load("pca_2_componentes.pkl")
# kmeans = joblib.load("kmeans_cancer.pkl")

# 11. Simular predicción con un nuevo dato (ejemplo ficticio)
nuevo_dato = X[0].reshape(1, -1)  # Tomamos un ejemplo del dataset original
# Supongamos que tienes un nuevo dato (30 características)
# nuevo_dato = np.array([[...]])
# nuevo_dato = np.array[[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
#                0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589,
#                153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
#                0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622,
#                0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

nuevo_escalado = scaler.transform(nuevo_dato)
nuevo_cluster = kmeans.predict(nuevo_escalado)
print(f"\nEl nuevo caso fue asignado al clúster: {nuevo_cluster[0]}")
