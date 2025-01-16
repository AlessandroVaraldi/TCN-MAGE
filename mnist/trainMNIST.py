import os
import gzip
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from utils.Trainer import Trainer
from models.tcn import TCN
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# === Configurazione dispositivo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === Seed per riproducibilità ===
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# === Funzione per estrarre i dati MNIST dai file gzip ===
def extract_mnist_gz(data_dir):
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    data = {}

    for file in files:
        path = os.path.join(data_dir, file)
        with gzip.open(path, 'rb') as f:
            if 'images' in file:
                data[file] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            elif 'labels' in file:
                data[file] = np.frombuffer(f.read(), np.uint8, offset=8)

    return data

# === Caricamento dataset MNIST ===
data_dir = "data/MNIST/raw"
data = extract_mnist_gz(data_dir)

# === Dataset personalizzato ===
class MNISTTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.data = torch.tensor(images, dtype=torch.float32) / 255.0  # Scala valori tra 0 e 1
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tratta ogni immagine 28x28 come sequenza di 28 passi temporali (colonne)
        x_sequence = self.data[idx].T  # Trasposta per avere colonne come passi temporali
        y_label = self.labels[idx]
        return x_sequence, y_label

train_images = data["train-images-idx3-ubyte.gz"]
train_labels = data["train-labels-idx1-ubyte.gz"]
test_images = data["t10k-images-idx3-ubyte.gz"]
test_labels = data["t10k-labels-idx1-ubyte.gz"]

mnist_dataset = MNISTTimeSeriesDataset(train_images, train_labels)

# === Divisione training, validation, test ===
train_size = int(0.7 * len(mnist_dataset))
val_size = int(0.15 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(mnist_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(MNISTTimeSeriesDataset(test_images, test_labels), batch_size=32, shuffle=False)

# === Configurazione modello ===
input_dim = 28  # 28 feature per time step
sequence_length = 28  # 28 time steps
num_classes = 10  # Cifre 0-9

config_path = "cfg/mnist_config.json"
# Configurazione esempio per TCN:
# {
#   "layers": [
#     {"hidden_dim": 64, "kernel_size": 3, "dilation": 1, "use_relu": true, "use_batchnorm": true},
#     {"hidden_dim": 128, "kernel_size": 3, "dilation": 2, "use_relu": true, "use_batchnorm": true},
#     {"hidden_dim": 256, "kernel_size": 3, "dilation": 4, "use_relu": true, "use_batchnorm": true}
#   ]
# }
tcn_model = TCN(input_dim=input_dim, config_path=config_path).to(device)

# === Training ===
model_path = f'checkpoints/best_tcn_mnist.pth'
epochs = 50
patience = 10

trainer = Trainer(tcn_model, device, learning_rate=1e-3)
tcn_model.train()
trained_model, train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)

# Salva modello
os.makedirs("checkpoints", exist_ok=True)
torch.save(trained_model.state_dict(), model_path)

# === Testing ===
trained_model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = trained_model(inputs)
        predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
        actuals.append(targets.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)
accuracy = np.mean(predictions == actuals)
print(f"\n> Test Accuracy: {accuracy:.4f}")

# Confusion Matrix and Classification Report
cm = confusion_matrix(actuals, predictions)
cmd = ConfusionMatrixDisplay(cm, display_labels=[f"{i}" for i in range(num_classes)])
cmd.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(actuals, predictions, target_names=[f"{i}" for i in range(num_classes)]))

# === Visualizzazione di un esempio ===
import random

# Estrai un esempio casuale dal dataset di test
random_idx = random.randint(0, len(test_dataset) - 1)
example_image, example_label = test_dataset[random_idx]
example_image_tensor = example_image.unsqueeze(0).to(device)  # Aggiungi batch dimension

with torch.no_grad():
    predicted_label = torch.argmax(trained_model(example_image_tensor)).item()

# Mostra l'immagine e la predizione
plt.imshow(example_image.T.cpu().numpy(), cmap="gray")  # Trasponi di nuovo per visualizzazione
plt.title(f"Label: {example_label}, Predicted: {predicted_label}")
plt.axis("off")
plt.show()
