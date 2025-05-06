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

"""## Ajuste de Hiperparametros"""

# Función de entrenamiento
def train_model(model, optimizer, criterion, train_loader, val_loader, device):
    model.to(device)
    for epoch in range(3):  # pocas épocas para pruebas rápidas
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.squeeze().numpy())

    return accuracy_score(targets, preds)

# Función objetivo para Optuna
def objective(trial):
    # 🔧 Hiperparámetros a optimizar
    conv1_out = trial.suggest_categorical('conv1_out', [16, 32, 64])
    conv2_out = trial.suggest_categorical('conv2_out', [32, 64, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    padding = trial.suggest_categorical('padding', [0, 1, 2])
    fc1_out = trial.suggest_categorical('fc1_out', [64, 128, 256])
    fc2_out = trial.suggest_categorical('fc2_out', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # DataLoaders con batch size optimizado
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Modelo con todos los hiperparámetros variables
    class TunedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, conv1_out, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=padding)
            self.pool = nn.MaxPool2d(2, 2)
            # Tamaño resultante tras 2 capas conv + 2 pooling (estimado para 28x28)
            out_size = 28
            out_size = (out_size + 2*padding - kernel_size) // 1 + 1
            out_size = out_size // 2
            out_size = (out_size + 2*padding - kernel_size) // 1 + 1
            out_size = out_size // 2
            self.flattened = conv2_out * out_size * out_size

            self.fc1 = nn.Linear(self.flattened, fc1_out)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(fc1_out, fc2_out)
            self.out = nn.Linear(fc2_out, n_clases)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, self.flattened)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.out(x)
            return x

    model = TunedCNN()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    accuracy = train_model(model, optimizer, criterion, train_loader, val_loader, device)
    return accuracy

# Configuración de entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)  # Puedes subir el número de pruebas

# Resultados
print("Mejores hiperparámetros encontrados:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

"""## Arquitectura final"""

# 1. Hiperparámetros optimizados
best_params = {
    'conv1_out': 64,
    'conv2_out': 128,
    'fc1_out': 256,
    'fc2_out': 64,
    'dropout': 0.282,
    'lr': 8.15e-4,
    'batch_size': 32
}

# 2. Creacion de modelo
class CNN_1C_Optimized(nn.Module):
    def __init__(self, n_clases):
        super(CNN_1C_Optimized, self).__init__()
        self.conv1 = nn.Conv2d(1, best_params['conv1_out'], kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(best_params['conv1_out'], best_params['conv2_out'], kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(best_params['dropout'])

        # Nueva función para calcular dimensiones automáticamente
        self._to_linear = None
        self._calculate_linear_size()

        self.fc1 = nn.Linear(self._to_linear, best_params['fc1_out'])
        self.fc2 = nn.Linear(best_params['fc1_out'], best_params['fc2_out'])
        self.fc3 = nn.Linear(best_params['fc2_out'], n_clases)

    def _calculate_linear_size(self):
        # Pasa un tensor dummy para calcular las dimensiones
        dummy_input = torch.zeros(1, 1, 28, 28)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        self._to_linear = dummy_output.numel() // dummy_output.size(0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)  # Usa el tamaño calculado automáticamente
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""## Preparacion de datos"""

# 4. Preparación del dataset (PathMNIST)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = PathMNIST(split='train', transform=transform, download=True)
val_dataset = PathMNIST(split='val', transform=transform, download=True)
test_dataset = PathMNIST(split="test", transform=transform, download=True)

# 5. Creación de DataLoaders optimizados
train_loader = DataLoader(train_dataset,
                        batch_size=best_params['batch_size'],
                        shuffle=True,
                        pin_memory=True if device.type == 'cuda' else False)

val_loader = DataLoader(val_dataset,
                        batch_size=best_params['batch_size'],
                        pin_memory=True if device.type == 'cuda' else False)

test_loader = DataLoader(test_dataset,
                        batch_size=best_params['batch_size'],
                        shuffle=False,  # ¡Importante para evaluación!
                        pin_memory=True if device.type == 'cuda' else False)

"""### Entrenamiento del modelo"""

# 6. Inicialización del modelo y optimizador
cnn_1c = CNN_1C_Optimized(n_clases=9).to(device)
optimizer = Adam(cnn_1c.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()

# 7. Funcion de entrenamiento
def train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.squeeze().long().to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validación
        val_acc = evaluate(model, val_loader, device)
        print(f'Época {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%')

# 8. Función de evaluación (val y test)
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 10. Entrenamiento
print("\nComenzando entrenamiento...")
train_model(cnn_1c, optimizer, criterion, train_loader, val_loader, device, epochs=10)

# 11. Evaluación final en Test
test_acc = evaluate(cnn_1c, test_loader, device)
print(f'\nAccuracy en Test: {test_acc:.2f}%')

# 12. Guardar modelo
torch.save(cnn_1c.state_dict(), 'modelo_optimizado_gpu.pth')
print("Modelo guardado correctamente")

"""# Segundo modelo"""

# Transforms (normalización de [0, 255] a [0.0, 1.0])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Cargar datasets
train_dataset = DataClass(split='train', transform=transform, download=True)
val_dataset = DataClass(split='val', transform=transform, download=True)
test_dataset = DataClass(split='test', transform=transform, download=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

"""## Arquitectura propuesta"""

class CNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

def evaluate(model, data_loader, x):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.squeeze().long().to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    conf_matrix = confusion_matrix(all_labels, all_preds)
    #print(f"Resultados: {x}")
    #plt.figure(figsize=(10, 10))
    #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    #plt.xlabel('Predicted')
    #plt.ylabel('True')
    #plt.show()
    print(f"Accuracy: {100 * correct / total:.2f}%")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(15):
  model = CNN(num_classes=9).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Entrenamiento simple
  for epoch in range(10):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
          images, labels = images.to(device), labels.squeeze().long().to(device)

          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

      print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
  evaluate(model, test_loader, "test")
  evaluate(model, val_loader, "validation")
