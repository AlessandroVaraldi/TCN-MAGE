import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import time
import os

# Verifica se CUDA è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === Parametro Fisso per Sequence Length ===
SEQUENCE_LENGTH = 8  # Imposta la lunghezza della sequenza in base alle tue esigenze

# === Imposta il seed per la riproducibilità ===
seed = 5
np.random.seed(seed)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_cols, output_cols, sequence_length=SEQUENCE_LENGTH):
        self.X = torch.tensor(data[input_cols].values, dtype=torch.float32)
        self.y = torch.tensor(data[output_cols].values, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        x_sequence = self.X[idx:idx + self.sequence_length].T
        y_value = self.y[idx + self.sequence_length - 1]
        return x_sequence, y_value

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        padding = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (padding, 0))
        out = self.conv(x)
        return out

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, kernel_size=3, num_layers=2):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList([CausalConv1d(input_dim, hidden_dim, kernel_size, dilation=1)])
        for i in range(1, num_layers - 1):
            dilation = 2 ** i
            self.layers.append(CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation))
        dilation = 2 ** (num_layers - 1)
        self.layers.append(CausalConv1d(hidden_dim, output_dim, kernel_size, dilation=dilation))
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            pre = layer(x)
            x = self.relu(pre)
        
        # Ultimo layer senza ReLU
        x = self.layers[-1](x)

        out = x[:, :, -1]
        return out

def check_and_clean_data(data, input_cols, output_cols):
    nan_count = data[input_cols + output_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"\n> Avviso: trovati {nan_count} valori NaN nel dataset. Verranno rimossi automaticamente.\n")
        data = data.dropna(subset=input_cols + output_cols).reset_index(drop=True)
    return data

def train_model(model, train_loader, val_loader, epochs=1000, patience=50, threshold=1e-4, learning_rate=1e-3):
    print(f"\n> Inizio training del modello: {type(model).__name__}\n")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    plt.ion()
    fig, ax = plt.subplots()

    best_model_path = f'best_model_{type(model).__name__}.pth'

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            ax.clear()
            ax.plot(train_losses, label='Training Loss')
            ax.plot(val_losses, label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.draw()
            plt.pause(0.01)

            if val_loss < best_val_loss - threshold:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"> Epoch {epoch+1}: Miglioramento della Validation Loss, modello salvato.")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\n> Early stopping: nessun miglioramento per {patience} epoche consecutive.\n")
                break
    except KeyboardInterrupt:
        print(f"\n> Training interrotto manualmente alla epoca {epoch+1}.\n")
    plt.ioff()
    plt.show()
    return model, train_losses, val_losses

def test_model(model, test_loader):
    model.eval()
    model.to(device)
    predictions, actuals = [], []
    start_inference_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())
    inference_time = time.time() - start_inference_time
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return predictions, actuals, inference_time

def calculate_rmse(predictions, actuals):
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    return rmse

def quantize_data(data, dtype, scale):
    scaled_data = data * scale
    return scaled_data.astype(dtype)

def save_quantized_data(data, path, dtypes, scales):
    for dtype, scale in zip(dtypes, scales):
        dtype_folder = os.path.join(path, dtype.__name__)
        os.makedirs(dtype_folder, exist_ok=True)
        quantized_data = quantize_data(data, dtype, scale)
        np.save(os.path.join(dtype_folder, "data.npy"), quantized_data)
        # Salva anche in formato binario
        with open(os.path.join(dtype_folder, "data.bin"), "wb") as f:
            f.write(quantized_data.tobytes())

def export_tcn_weights(model, output_dir, scales):
    dtypes = [np.float32, np.int32, np.int16, np.int8]

    for i, layer in enumerate(model.layers):
        layer_dir = os.path.join(output_dir, f"layer{i}")
        os.makedirs(layer_dir, exist_ok=True)

        w = layer.conv.weight.data.cpu().numpy()
        b = layer.conv.bias.data.cpu().numpy()

        weight_dir = os.path.join(layer_dir, "weights")
        bias_dir = os.path.join(layer_dir, "biases")
        os.makedirs(weight_dir, exist_ok=True)
        os.makedirs(bias_dir, exist_ok=True)

        save_quantized_data(w, weight_dir, dtypes, scales)
        save_quantized_data(b, bias_dir, dtypes, scales)

# Parametri predefiniti
file_path = "NYC_Weather_2016_2022.csv"
input_cols = ["temperature_2m (°C)", "cloudcover (%)", "cloudcover_low (%)", "cloudcover_mid (%)", "cloudcover_high (%)", "precipitation (mm)", "winddirection_10m (°)"]
input_cols = ["temperature_2m (°C)", "cloudcover (%)", "precipitation (mm)", "winddirection_10m (°)"]
output_cols = ["windspeed_10m (km/h)"]

# Caricamento e pulizia del dataset
data = pd.read_csv(file_path)
data = check_and_clean_data(data, input_cols, output_cols)

dataset = TimeSeriesDataset(data, input_cols, output_cols)
train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Parametri per il modello TCN
input_dim = len(input_cols)
output_dim = len(output_cols)

hidden_dim = 2
kernel_size = 3 
num_layers = 3

tcn_model = TCN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)

# Addestramento modello TCN
trained_tcn, _, _ = train_model(tcn_model, train_loader, val_loader, epochs=100, patience=100, learning_rate=1e-3)

# Test del modello TCN
tcn_predictions, tcn_actuals, tcn_inf_time = test_model(trained_tcn, test_loader)
tcn_rmse = calculate_rmse(tcn_predictions, tcn_actuals)

print(f"\n> RMSE TCN: {tcn_rmse:.4f}")
print(f"\n> Tempo di inferenza TCN: {tcn_inf_time:.4f} secondi")

plt.figure()
plt.plot(tcn_actuals, label="Actual")
plt.plot(tcn_predictions, label="TCN Predictions")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()
plt.title("Predictions TCN")
plt.show()

# Esportazione modello TCN in ONNX
dummy_input = torch.randn(1, input_dim, SEQUENCE_LENGTH).to(device)
torch.onnx.export(trained_tcn, dummy_input, "tcn_model.onnx", verbose=True)

# Ricarichiamo il best model TCN
best_tcn = TCN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)
best_tcn.load_state_dict(torch.load("best_model_TCN.pth", map_location=device))
best_tcn.eval()

# Esporta i pesi per il codice C
scales = [1.0, (2**16), (2**8), (2**4)]
export_tcn_weights(best_tcn, "data", scales)

# Ora creiamo un input di test e calcoliamo l'output di riferimento
batch = 1

torch_input = torch.randn(batch, input_dim, SEQUENCE_LENGTH, dtype=torch.float32)
with torch.no_grad():
    py_output = best_tcn(torch_input).cpu().numpy() # [batch, output_dim]

print("Python output di riferimento:", py_output)

# Salviamo l'input e l'output di riferimento in binario
input_dir = os.path.join("data", "inputs")
output_dir = os.path.join("data", "outputs")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

save_quantized_data(torch_input.cpu().numpy(), input_dir, [np.float32, np.int32, np.int16, np.int8], scales)
save_quantized_data(py_output, output_dir, [np.float32, np.int32, np.int16, np.int8], scales)

print("> Input e output di riferimento esportati con successo!")
