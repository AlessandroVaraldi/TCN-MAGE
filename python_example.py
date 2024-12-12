import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Funzione per generare il deterministico input (sinusoide)
def generate_deterministic_input(sequence_length, input_channels):
    x = np.linspace(0, 2 * np.pi, sequence_length)
    input_sequence = np.sin(x).reshape(-1, 1)
    return torch.tensor(input_sequence, dtype=torch.float32).view(1, input_channels, sequence_length)

# Definizione della rete TCN in PyTorch
class TCN(nn.Module):
    def __init__(self, num_layers, input_channels, output_channels, kernel_size=3):
        super(TCN, self).__init__()
        self.num_layers = num_layers
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            dilation = 2 ** i
            out_channels = output_channels if i == num_layers - 1 else in_channels * 2
            padding = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding))
            layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(in_channels, output_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.network(x)
        x = self.final_conv(x)
        return x

# Parametri di rete
num_layers = 8
input_channels = 1
output_channels = 16
sequence_length = 1024
learning_rate = 0.001
num_epochs = 5000

# Generazione dell'input deterministico
input_sequence = generate_deterministic_input(sequence_length, input_channels)

# Inizializzazione del modello TCN
model = TCN(num_layers, input_channels, output_channels)
print(model)

# Definizione della funzione di loss (MSE) e dell'ottimizzatore
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Output desiderato (la sequenza successiva della sinusoidale)
output_sequence = torch.sin(torch.linspace(0, 2 * np.pi, sequence_length).reshape(1, output_channels, sequence_length))

# Training del modello
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(input_sequence)
    loss = criterion(output, output_sequence)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# Visualizzazione dei risultati
model.eval()
with torch.no_grad():
    predicted_output = model(input_sequence).squeeze().numpy()
    plt.plot(predicted_output, label='Predicted Output')
    plt.plot(output_sequence.squeeze().numpy(), label='True Output', linestyle='dashed')
    plt.legend()
    plt.title("TCN Output vs True Output")
    plt.show()
