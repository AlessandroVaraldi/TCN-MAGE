import torch
import torch.optim as optim
import torch.nn as nn
from tcn_model import TCN
import numpy as np
import math
import matplotlib.pyplot as plt

# Funzione per generare dati deterministici di vibrazioni
def generate_vibrations(batch_size, sequence_length, input_channels, seed=42):
    np.random.seed(seed)  # Imposta il seed per la riproducibilità
    data = np.zeros((batch_size, input_channels, sequence_length))
    
    for b in range(batch_size):
        for c in range(input_channels):
            # Parametri variabili per ogni canale
            frequency = np.random.uniform(0.1, 0.5)  # Frequenza tra 0.1 e 0.5 Hz
            amplitude = np.random.uniform(0.5, 1.5)  # Ampiezza tra 0.5 e 1.5
            phase = np.random.uniform(0, 2 * np.pi)  # Fase casuale per ogni canale
            
            # Genera vibrazioni utilizzando una funzione sinusoidale
            for t in range(sequence_length):
                time = t / float(sequence_length)
                # Formula per vibrazioni (somma di sinusoidi)
                data[b, c, t] = amplitude * math.sin(2 * np.pi * frequency * time + phase)
                
    return data

# Funzione per convertire i dati in un tensore torch
def generate_data(batch_size, sequence_length, input_channels, seed=42):
    data = generate_vibrations(batch_size, sequence_length, input_channels, seed)
    # Convertiamo in tensori torch
    data = torch.tensor(data, dtype=torch.float32)
    return data

# Funzioni per generare i dati di training, validazione e test
def generate_validation_data(batch_size, sequence_length, input_channels, seed=42):
    return generate_data(batch_size, sequence_length, input_channels, seed)

def generate_test_data(batch_size, sequence_length, input_channels, seed=42):
    return generate_data(batch_size, sequence_length, input_channels, seed)

def train():
    # Parametri del modello
    sequence_length = 100
    batch_size = 32
    num_epochs = 1000
    
    # Configurazione personalizzata dei layer
    layers_config = [
        {'input_channels': 1, 'output_channels': 16, 'kernel_size': 3, 'dilation': 1},
        {'input_channels': 16, 'output_channels': 32, 'kernel_size': 3, 'dilation': 2},
        {'input_channels': 32, 'output_channels': 64, 'kernel_size': 3, 'dilation': 4},
        {'input_channels': 64, 'output_channels': 128, 'kernel_size': 3, 'dilation': 8},
        {'input_channels': 128, 'output_channels': 64, 'kernel_size': 3, 'dilation': 16},
        {'input_channels': 64, 'output_channels': 32, 'kernel_size': 3, 'dilation': 32},
        {'input_channels': 32, 'output_channels': 16, 'kernel_size': 3, 'dilation': 64},
        {'input_channels': 16, 'output_channels': 1, 'kernel_size': 3, 'dilation': 128},
    ]
    
    # Crea il modello TCN
    model = TCN(layers_config, sequence_length)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ottimizzatore e funzione di perdita
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Genera i dati di validazione e test separati
    validation_data = generate_validation_data(batch_size, sequence_length, layers_config[0]['input_channels'], seed=43)
    test_data = generate_test_data(batch_size, sequence_length, layers_config[0]['input_channels'], seed=44)

    validation_data = validation_data.to('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = test_data.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ciclo di addestramento
    for epoch in range(num_epochs):
        model.train()
        
        # Genera i dati di training
        X_train = generate_data(batch_size, sequence_length, layers_config[0]['input_channels'], seed=42)
        noise = torch.randn_like(X_train) * 0.1  # Aggiungiamo del rumore
        y_train = X_train + noise  # y è X con del rumore
        
        X_train, y_train = X_train.to('cuda' if torch.cuda.is_available() else 'cpu'), y_train.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Forward pass sui dati di training
        optimizer.zero_grad()
        outputs_train = model(X_train)
        outputs_train = outputs_train[:, :, -100:]  # Usa solo gli ultimi 100 valori
        
        # Calcola la perdita sui dati di training
        loss_train = criterion(outputs_train, y_train)
        
        # Backpropagation e aggiornamento dei pesi
        loss_train.backward()
        optimizer.step()
        
        # Valutazione sui dati di validazione
        model.eval()
        with torch.no_grad():
            outputs_val = model(validation_data)
            loss_val = criterion(outputs_val[:, :, -100:], validation_data + noise)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss_train.item()}, Validation Loss: {loss_val.item()}')
        
        if epoch % 100 == 0:  # Plotta ogni 100 epoche
            # Visualizza il risultato per un batch di validazione
            plt.figure(figsize=(10, 6))
            idx = 0

            # Dati originali (y)
            plt.plot(np.arange(sequence_length), validation_data[idx, 0, :].cpu().numpy(), label="Target (y)", linestyle='--')

            # Output del modello (outputs)
            plt.plot(np.arange(sequence_length), outputs_val[idx, 0, :].cpu().detach().numpy(), label="Prediction (outputs)", linestyle='-')

            plt.title(f"Epoch {epoch+1}: Target vs Prediction on Validation Data")
            plt.xlabel("Time Step")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Valutazione finale sui dati di test
    model.eval()
    with torch.no_grad():
        outputs_test = model(test_data)
        loss_test = criterion(outputs_test[:, :, -100:], test_data + noise)
        
    print(f"Final Test Loss: {loss_test.item()}")
    
    # Visualizza una previsione finale sui dati di test
    plt.figure(figsize=(10, 6))
    idx = 0

    # Dati originali (y)
    plt.plot(np.arange(sequence_length), test_data[idx, 0, :].cpu().numpy(), label="Target (y)", linestyle='--')

    # Output del modello (outputs)
    plt.plot(np.arange(sequence_length), outputs_test[idx, 0, :].cpu().detach().numpy(), label="Prediction (outputs)", linestyle='-')

    plt.title("Final Prediction vs Target on Test Data")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
