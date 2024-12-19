import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
from torch.utils.data import DataLoader, TensorDataset, random_split

# Definizione della TCN con gestione corretta del padding
class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, layer_configs):
        super(TemporalConvolutionalNetwork, self).__init__()

        self.layers = nn.ModuleList()
        
        # Creazione dei layer con i parametri specificati
        for config in layer_configs:
            padding = (config['kernel_size'] - 1) * config['dilation'] // 2
            layer = nn.Conv1d(
                in_channels=config['input_size'],
                out_channels=config['output_size'],
                kernel_size=config['kernel_size'],
                dilation=config['dilation'],
                padding=padding
            )
            self.layers.append(layer)
            self.layers.append(nn.ReLU())

        self.final_layer = nn.Conv1d(
            in_channels=layer_configs[-1]['output_size'],
            out_channels=1,
            kernel_size=1
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_layer(x)
        return x

# Funzione di training con salvataggio del modello migliore
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, model_save_path="best_model.pth"):
    best_val_loss = float('inf')
    best_model_wts = None
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Training
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        val_loss = 0
        model.eval()  # Passa in modalità di valutazione per il calcolo sulla validation
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        
        # Salva il modello se la validation loss è migliorata
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, model_save_path)
            print(f"Saving model with improved validation loss: {avg_val_loss}")

# Funzione di inferenza
def infer(model, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output

# Funzione per fermare il training con una combinazione di tasti
def signal_handler(sig, frame):
    print('Training interrotto manualmente.')
    sys.exit(0)

# Impostazione del signal handler per fermare il training
signal.signal(signal.SIGINT, signal_handler)

# Funzione per il plotting dell'inferenza su un solo grafico
def plot_inference(input_data, output_data):
    plt.figure(figsize=(10, 6))
    
    # Traccia l'input e l'output sullo stesso grafico
    plt.plot(input_data.numpy().flatten(), label="Input", color='blue')
    plt.plot(output_data.numpy().flatten(), label="Output", color='red', linestyle='--')
    
    # Aggiungi titolo e etichette
    plt.title("Confronto tra Input e Output")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    
    # Aggiungi una legenda
    plt.legend()
    
    # Mostra il grafico
    plt.show()

# Funzione per calcolare il RMSE
def calculate_rmse(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    return rmse

# Parametrizzazione dei layer
layer_configs = [
    {'input_size': 1, 'output_size': 32, 'kernel_size': 3, 'dilation': 1},  # Primo layer
    {'input_size': 32, 'output_size': 64, 'kernel_size': 3, 'dilation': 2},  # Secondo layer
    {'input_size': 64, 'output_size': 128, 'kernel_size': 3, 'dilation': 4},  # Terzo layer
    {'input_size': 128, 'output_size': 64, 'kernel_size': 3, 'dilation': 8},  # Quarto layer
    {'input_size': 64, 'output_size': 32, 'kernel_size': 3, 'dilation': 16},  # Quinto layer
    {'input_size': 32, 'output_size': 1, 'kernel_size': 1, 'dilation': 1}  # Layer finale
]

# Generazione dei dati di esempio
num_samples = 1000
sequence_length = 50  # Lunghezza della sequenza (comune per X e y)

X = np.random.randn(num_samples, sequence_length, 1)  # 1 canale di input
y = np.random.randn(num_samples, sequence_length, 1)  # 1 canale di output

# Conversione in tensori PyTorch e permutazione per adattamento a Conv1d
X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (batch, channel, sequence)
y = torch.tensor(y, dtype=torch.float32).permute(0, 2, 1)  # (batch, channel, sequence)

# Creazione del DataLoader
dataset = TensorDataset(X, y)

# Suddivisione del dataset in training, validation e test (70% - 15% - 15%)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inizializzazione del modello
model = TemporalConvolutionalNetwork(layer_configs)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Numero di epoche per il training
epochs = 100

# Avvio del training
model_save_path = "best_model.pth"
try:
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, model_save_path=model_save_path)
except KeyboardInterrupt:
    pass

# Carica il modello con la migliore performance sulla validation
model.load_state_dict(torch.load(model_save_path))

# Inferenza sul dataset di test
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        predictions = model(data)
        all_predictions.append(predictions)
        all_targets.append(target)

# Concatenare tutte le previsioni e i target
all_predictions = torch.cat(all_predictions, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Visualizzazione dell'inferenza sui dati di test (primo esempio)
plot_inference(all_targets[0], all_predictions[0])

# Calcolo del RMSE sul dataset di test
rmse = calculate_rmse(all_targets, all_predictions)
print(f"RMSE sul dataset di test: {rmse.item()}")
