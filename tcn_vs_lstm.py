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

# Verifica se CUDA è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === Parametro Fisso per Sequence Length ===
SEQUENCE_LENGTH = 8  # Imposta la lunghezza della sequenza in base alle tue esigenze

# === Classe per la creazione del dataset ===
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_cols, output_cols, sequence_length=SEQUENCE_LENGTH):
        self.X = torch.tensor(data[input_cols].values, dtype=torch.float32)
        self.y = torch.tensor(data[output_cols].values, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        x_sequence = self.X[idx:idx + self.sequence_length].T  # Shape (num_features, sequence_length)
        y_value = self.y[idx + self.sequence_length - 1]  # Prendiamo l'ultimo valore come target
        return x_sequence, y_value


# === CausalConv1d: Classe per la convoluzione causale 1D ===
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)

    def forward(self, x):
        padding = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (padding, 0))
        return self.conv(x)


# === Modello TCN ===
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList([CausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=1)])
        for i in range(1, num_layers - 1):
            dilation = 2 ** i
            self.layers.append(CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation))
        dilation = 2 ** (num_layers - 1)
        self.layers.append(CausalConv1d(hidden_dim, output_dim, kernel_size=3, dilation=dilation))
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x[:, :, -1]


# === Modello LSTM ===
class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])
    
def load_data():
    while True:
        file_path = input("\n> Inserisci il percorso del file .csv (ad esempio, 'dati.csv'): ")
        try:
            data = pd.read_csv(file_path)
            print("\n> Colonne disponibili nel dataset:", list(data.columns))
            input_cols = input("\n> Seleziona le colonne di input (separate da virgola): ").split(",")
            output_cols = input("\n> Seleziona le colonne di output (separate da virgola): ").split(",")

            # Controllo colonne
            missing_cols = [col for col in input_cols + output_cols if col not in data.columns]
            if missing_cols:
                print(f"\n> Errore: le seguenti colonne non sono presenti nel dataset: {missing_cols}\n")
                continue

            return data, input_cols, output_cols
        except FileNotFoundError:
            print("\n> Errore: il file specificato non è stato trovato. Riprova.\n")
        except pd.errors.EmptyDataError:
            print("\n> Errore: il file è vuoto o danneggiato. Riprova.\n")


# === Funzione per la pulizia dei dati e la gestione dei NaN ===
def check_and_clean_data(data, input_cols, output_cols):
    nan_count = data[input_cols + output_cols].isna().sum().sum()
    if nan_count > 0:
        print(f"\n> Avviso: trovati {nan_count} valori NaN nel dataset. Verranno rimossi automaticamente.\n")
        data = data.dropna(subset=input_cols + output_cols).reset_index(drop=True)
    return data


# === Funzione di training ===
def train_model(model, train_loader, val_loader, epochs=1000, patience=50, threshold=1e-4, learning_rate=1e-3):
    print(f"\n> Inizio training del modello: {type(model).__name__}\n")
    model.to(device)  # Trasferisce il modello su GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    plt.ion()
    fig, ax = plt.subplots()

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Trasferisce i dati su GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)  # Trasferisce i dati su GPU
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Aggiornamento grafico
            ax.clear()
            ax.plot(train_losses, label='Training Loss')
            ax.plot(val_losses, label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.draw()
            plt.pause(0.01)

            # Salvataggio miglior modello
            if val_loss < best_val_loss - threshold:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), f'best_model_{type(model).__name__}.pth')
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


# === Funzione per testare il modello ===
def test_model(model, test_loader):
    model.eval()
    model.to(device)  # Trasferisce il modello su GPU
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())  # Riporta i dati su CPU per elaborazioni successive
            actuals.append(targets.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return predictions, actuals


# === Funzione per il calcolo dell'RMSE ===
def calculate_rmse(predictions, actuals):
    return np.sqrt(np.mean((predictions - actuals) ** 2))


# === Funzione per confrontare TCN e LSTM ===
def compare_models(tcn_model, lstm_model, train_loader, val_loader, test_loader, epochs=100, patience=10, learning_rate=1e-3):
    print("\n> Inizio confronto dei modelli...")

    # Addestramento TCN
    print("\n> Addestramento modello TCN...")
    trained_tcn, _, _ = train_model(tcn_model, train_loader, val_loader, epochs, patience, learning_rate)

    # Addestramento LSTM
    print("\n> Addestramento modello LSTM...")
    trained_lstm, _, _ = train_model(lstm_model, train_loader, val_loader, epochs, patience, learning_rate)

    # Test modelli
    print("\n> Test modelli...")
    tcn_predictions, tcn_actuals = test_model(trained_tcn, test_loader)
    lstm_predictions, lstm_actuals = test_model(trained_lstm, test_loader)

    # Calcolo RMSE
    tcn_rmse = calculate_rmse(tcn_predictions, tcn_actuals)
    lstm_rmse = calculate_rmse(lstm_predictions, lstm_actuals)

    print(f"\n> RMSE TCN: {tcn_rmse:.4f}")
    print(f"> RMSE LSTM: {lstm_rmse:.4f}")

    # Confronto visivo
    plt.figure()
    plt.plot(tcn_actuals, label="Actual")
    plt.plot(tcn_predictions, label="TCN Predictions")
    plt.plot(lstm_predictions, label="LSTM Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Output")
    plt.legend()
    plt.title("Confronto tra TCN e LSTM")
    plt.show()


# === Funzione per calcolare il tempo di inferenza ===
def measure_inference_time(model, data_loader, num_batches=10):
    model.eval()
    model.to(device)  # Trasferisce il modello su GPU
    total_time = 0.0
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            if i >= num_batches:  # Limita il numero di batch
                break
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
    avg_time_per_batch = total_time / min(len(data_loader), num_batches)
    return avg_time_per_batch


# === Confronto dei tempi di inferenza ===
def compare_inference_time(tcn_model, lstm_model, test_loader):
    print("\n> Calcolo del tempo medio di inferenza...")
    tcn_time = measure_inference_time(tcn_model, test_loader)
    lstm_time = measure_inference_time(lstm_model, test_loader)

    print(f"\n> Tempo medio di inferenza per batch:")
    print(f"  - TCN: {tcn_time:.6f} secondi")
    print(f"  - LSTM: {lstm_time:.6f} secondi")


# === Caricamento del Dataset ===
print("\n=== Caricamento del Dataset e Selezione delle Colonne ===")
data, input_cols, output_cols = load_data()
data = check_and_clean_data(data, input_cols, output_cols)

# Creazione e suddivisione del dataset
dataset = TimeSeriesDataset(data, input_cols, output_cols)
train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Parametri per i modelli
input_dim, output_dim, hidden_dim, num_layers = len(input_cols), len(output_cols), 16, 3
tcn_model = TCN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
lstm_model = LSTMModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)

# Confronto modelli
compare_models(tcn_model, lstm_model, train_loader, val_loader, test_loader, epochs=100, patience=100, learning_rate=1e-3)

# Esportazione modelli in ONNX
dummy_input = torch.randn(1, input_dim, SEQUENCE_LENGTH).to(device)
torch.onnx.export(tcn_model, dummy_input, "tcn_model.onnx", verbose=True)
torch.onnx.export(lstm_model, dummy_input, "lstm_model.onnx", verbose=True)

# Confronto dei tempi di inferenza
print("\n=== Confronto dei Tempi di Inferenza ===")
compare_inference_time(tcn_model, lstm_model, test_loader)


# path: NYC_Weather_2016_2022.csv
# output: windspeed_10m (km/h)
# input: temperature_2m (°C),cloudcover (%),cloudcover_low (%),cloudcover_mid (%),cloudcover_high (%),precipitation (mm),winddirection_10m (°)