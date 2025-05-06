import torch

#Verificar si estamos usando la GPU
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

"""# Bibliotecas"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score

import optuna
from medmnist import PathMNIST, INFO

import matplotlib.pyplot as plt
import numpy as np

"""# EDA

## Importación de datos
"""

info = INFO["pathmnist"]
n_clases = len(info["label"])

#Carga de datos
visualizar = PathMNIST(split= "train", download=True)
labels = visualizar.labels

"""## Numero de muestras por clase"""

#Imprimir numero de muestras en cada clase
clas_name = [
    "ADI (Adipose)",         # Tejido adiposo (Grasa)
    "BACK (Background)",     # Fondo sin tejido relevante (espacios vacios)
    "DEB (Debris)",          # Restos celulares o tejido degradado
    "LYM (Lymphocytes)",     # Inflirtados linfocitos (celulas inmunes)
    "MUC (Mucus)",           # Mucosa o sececiones
    "MUS (Smooth muscle)",   # Musculo liso
    "NORM (Normal mucosa)",  # Tejido epitelial normal (mucosa sana)
    "STR (Stroma)",          # *Estroma tumoral (tejido conectivo alrededor de tumores)
    "TUM (Tumor epithelium)" # *Epitelio tumoral (tejido canceroso)
]
unique_clases, counts = np.unique(labels, return_counts=True)

print("Distribuciones de clases - PathMNIST (train)")
for cls, count, name in zip(unique_clases, counts, clas_name):
    print(f"Clase {cls}: {name:<22} -> {count} muestras")

# Imprimir distribuciones de las clases de manera visual

# Creacion de colores con degradado
plt.figure(figsize=(12,5))
plt.bar(clas_name, counts, color = "Purple")
plt.xlabel("Clase")
plt.ylabel("Numero de muestas")
plt.title("Distribuciones de clase - PathMNIST (train)")
plt.xticks(rotation = 45, ha = "right")

plt.tight_layout
plt.show()

"""## Ejemplos de las clases"""

fig, axes = plt.subplots(3, 3, figsize=(9, 8))
for cls in range(9):
    idx = np.where(labels == cls)[0][0]  # Primera imagen de la clase
    img, _ = visualizar[idx]
    axes[cls//3, cls%3].imshow(img, cmap="gray")
    axes[cls//3, cls%3].set_title(f"{cls} - {clas_name[cls]}")
    axes[cls//3, cls%3].axis('off')
plt.tight_layout()
plt.show()

"""## Prueba χ2"""

"""Para determinar si la clase con más muestras es significativamente diferente a las demás
(y justificar el aumento de datos), realizaremos una prueba estadística  chi-cuadrado (χ2).

-> Hipótesis nula (H0): No hay diferencia significativa entre la proporción de muestras de la clase mayoritaria y las demás.
-> Hipótesis alternativa (H1): La clase mayoritaria tiene una proporción significativamente mayor.

Si rechazamos H0, concluiremos que el desbalance es estadísticamente significativo y justificando el aumento de datos."""

from scipy.stats import chisquare

# Carga de datos
prueba = PathMNIST(split="train", download=True)
label_prueba = prueba.labels
unique_clases_prueba, counts_prueba = np.unique(labels, return_counts=True)


observed = counts_prueba #  observada
expected = np.full_like(observed, fill_value= np.mean(observed)/len(observed)) # Frecuencia esperada
expected = expected * (np.sum(observed)/ np.sum(expected))

# Prueba Chi2
chi2_stat, p_valor = chisquare(f_obs=observed, f_exp=expected)

print(f"Estadistico χ2: {chi2_stat:4f}")
print(f"p_valor: {p_valor:4f}")

alpha = 0.05
if p_valor < alpha:
    print("Rechazamos H0, Hay diferencia significativa")
else:
    print("No rechazamos H0, No hay diferencia significativa")


# Viendo que hay diferencia significativa haremos modelos con aumento de datos

"""# CNN-1C"""

# Transformar datos a escala de grises (1 Canal)
conversor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

#Carga de datos
train_dataset = PathMNIST(split="train", transform=conversor, download=True)
val_dataset = PathMNIST(split="val", transform=conversor, download=True)
test_dataset = PathMNIST(split="test", transform=conversor, download=True)

train_loader = DataLoader(train_dataset, batch_size=64 , shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64 , shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64 , shuffle=True)

"""## Propuesta de arquitectura"""

"""Modelo convoluciona1- 1 canal"""
class CNN_1C(nn.Module):
    def __init__(self):
        super(CNN_1C).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 7 * 7,128)
        self.fc2 = nn.Linear(128, n_clases)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
