# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 12:05:28 2025

@author: IVAN
"""

# Crear un ejemplo visual tipo "foto original vs. reconstruida (reducción de información PCA)

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Usar el dataset de dígitos (imágenes de 8x8)
digits = load_digits()
X = digits.data
images = digits.images

# Aplicar PCA para reducir dimensiones (por ejemplo, a 10 componentes, ó con 2)
pca_digits = PCA(n_components=10)
X_pca_digits = pca_digits.fit_transform(X)
X_reconstruido = pca_digits.inverse_transform(X_pca_digits)

# Mostrar 10 imágenes originales y reconstruidas
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for i in range(10):
    # Imagen original
    axes[0, i].imshow(images[i], cmap='gray')
    axes[0, i].axis('off')
    # Imagen reconstruida (reconvertida desde 10 componentes)
    axes[1, i].imshow(X_reconstruido[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstruida (PCA)")

plt.subplots_adjust(hspace=1.5) # espacio vertical extra
plt.suptitle("Ejemplo visual: Varianza explicada con 10 componentes")
plt.tight_layout()
plt.show()
