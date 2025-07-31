# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:54:40 2025

@author: IVAN
"""

# Entrena un modelo K-NN con y sin PCA.
# Guarda el modelo y carga para predicción de nuevos casos.

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Cargar dataset
data = load_breast_cancer()
X = data.data # type: ignore
y = data.target # type: ignore

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PCA con 10 componentes
# PCA se aplica después del preprocesamiento y EDA, justo antes del modelado.
# PCA debe aplicarse solo sobre el training set
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# K-NN sin PCA
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_no_pca = accuracy_score(y_test, y_pred)
f1_no_pca = f1_score(y_test, y_pred)

# K-NN con PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)

print({
    "Exactitud sin PCA": acc_no_pca,
    "F1-score sin PCA": f1_no_pca,
    "Exactitud con PCA (3 componentes)": acc_pca,
    "F1-score con PCA (3 componentes)": f1_pca
})


# Predecir un nuevo dato
# Supongamos que tienes un nuevo dato (con las mismas 30 características):
nuevo_dato = [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
               0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589,
               153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
               0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622,
               0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

# 1. Escalar con el scaler entrenado
nuevo_dato_escalado = scaler.transform(nuevo_dato)

# 2. Aplicar PCA con el transformador ya entrenado
nuevo_dato_pca = pca.transform(nuevo_dato_escalado)

# 3. Predecir con el modelo entrenado con PCA
prediccion = knn_pca.predict(nuevo_dato_pca)

print("Predicción:", "Benigno" if prediccion[0] == 1 else "Maligno")



# Guardar el modelo

import joblib

# Guardar scaler, PCA y modelo KNN con PCA
joblib.dump(scaler, "escalador.pkl")
joblib.dump(pca, "pca_3_componentes.pkl")
joblib.dump(knn_pca, "knn_pca_modelo.pkl")


# Cargar el modelo
import joblib

# Cargar modelos previamente guardados
scaler = joblib.load("escalador.pkl")
pca = joblib.load("pca_componentes.pkl")
modelo = joblib.load("knn_pca_modelo.pkl")

# Nuevo dato (con las 30 características originales)
# Reemplaza con tus valores
nuevo_dato = [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
               0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589,
               153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
               0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622,
               0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

# Preprocesar y predecir un nuevo dato
dato_escalado = scaler.transform(nuevo_dato)
dato_pca = pca.transform(dato_escalado)
prediccion = modelo.predict(dato_pca)

print("Predicción (load):", "Benigno" if prediccion[0] == 1 else "Maligno")
