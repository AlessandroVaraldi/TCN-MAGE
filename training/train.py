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
SEQUENCE_LENGTH = 32  # Imposta la lunghezza della sequenza

# === Imposta il seed per la riproducibilità ===
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

# --- Dataset Utilities ---
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

class DataUtils:
    @staticmethod
    def check_and_clean_data(data, input_cols, output_cols):
        nan_count = data[input_cols + output_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"\n> Avviso: trovati {nan_count} valori NaN nel dataset. Verranno rimossi automaticamente.\n")
            data = data.dropna(subset=input_cols + output_cols).reset_index(drop=True)
        return data

# --- Model Definitions ---
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
            x = self.relu(layer(x))
        return self.layers[-1](x)[:, :, -1]

# --- Training Utilities ---
class Trainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device

    def train(self, train_loader, val_loader, epochs=1000, patience=50, threshold=1e-4):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = f'best_model_{type(self.model).__name__}.pth'

        plt.ion()
        fig, ax = plt.subplots()

        try:
            for epoch in range(epochs):
                train_loss = self._train_one_epoch(train_loader)
                val_loss = self._validate_one_epoch(val_loader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                self._update_plot(ax, train_losses, val_losses, epoch)

                if val_loss < best_val_loss - threshold:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), best_model_path)
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
        return self.model, train_losses, val_losses

    def _train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def _validate_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, targets).item()
        return total_loss / len(val_loader)

    def _update_plot(self, ax, train_losses, val_losses, epoch):
        ax.clear()
        ax.plot(train_losses, label='Training Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.annotate(f'Epoch {epoch}', xy=(0, 0), xycoords='axes fraction', ha='left', va='bottom')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.draw()
        plt.pause(0.01)

# --- Testing Utilities ---
class Tester:
    @staticmethod
    def test_model(model, test_loader, device):
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
        return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0), inference_time

    @staticmethod
    def calculate_rmse(predictions, actuals):
        return np.sqrt(np.mean((predictions - actuals) ** 2))

# --- Export Utilities ---
class Exporter:
    @staticmethod
    def quantize_data(data, dtype, scale):
        scaled_data = data * scale
        return scaled_data.astype(dtype)

    @staticmethod
    def save_quantized_data(data, path, dtypes, scales):
        for dtype, scale in zip(dtypes, scales):
            dtype_folder = os.path.join(path, dtype.__name__)
            os.makedirs(dtype_folder, exist_ok=True)
            quantized_data = Exporter.quantize_data(data, dtype, scale)
            np.save(os.path.join(dtype_folder, "data.npy"), quantized_data)
            with open(os.path.join(dtype_folder, "data.bin"), "wb") as f:
                f.write(quantized_data.tobytes())

    @staticmethod
    def export_tcn_weights(model, output_dir, scales):
        dtypes = [np.float32, np.int32, np.int16, np.int8]
        for i, layer in enumerate(model.layers):
            layer_dir = os.path.join(output_dir, f"layer{i}")
            os.makedirs(layer_dir, exist_ok=True)
            w = layer.conv.weight.data.cpu().numpy()
            b = layer.conv.bias.data.cpu().numpy()
            Exporter.save_quantized_data(w, os.path.join(layer_dir, "weights"), dtypes, scales)
            Exporter.save_quantized_data(b, os.path.join(layer_dir, "biases"), dtypes, scales)

    @staticmethod
    def export_reference_input_output(input_data, output_data, header_path, dtype):
        with open(header_path, 'w') as f:
            f.write(f"#ifndef TCN_IO_REF_{dtype.upper()}_H\n")
            f.write(f"#define TCN_IO_REF_{dtype.upper()}_H\n\n")
            f.write("#include <stdint.h>\n\n")

            c_type = {
                'float32': 'float',
                'int32': 'int32_t',
                'int16': 'int16_t',
                'int8': 'int8_t'
            }[dtype]

            flat_input = input_data.flatten()
            f.write(f"static const {c_type} INPUT_REF[{input_data.size}] = {{\n")
            for i, value in enumerate(flat_input):
                if i % 8 == 0 and i > 0:
                    f.write("\n    ")
                f.write(f"{value}, ")
            f.write("\n};\n\n")

            flat_output = output_data.flatten()
            f.write(f"static const {c_type} OUTPUT_REF[{output_data.size}] = {{\n")
            for i, value in enumerate(flat_output):
                if i % 8 == 0 and i > 0:
                    f.write("\n    ")
                f.write(f"{value}, ")
            f.write("\n};\n\n")

            f.write(f"#endif // TCN_IO_REF_{dtype.upper()}_H\n")

    @staticmethod
    def generate_combined_header_file(data, header_path, dtype_name):
        with open(header_path, 'w') as f:
            f.write(f"#ifndef TCN_WEIGHTS_BIASES_{dtype_name.upper()}_H\n")
            f.write(f"#define TCN_WEIGHTS_BIASES_{dtype_name.upper()}_H\n\n")
            f.write("#include <stdint.h>\n\n")

            c_type = {
                'float32': 'float',
                'int32': 'int32_t',
                'int16': 'int16_t',
                'int8': 'int8_t'
            }[dtype_name]

            f.write(f"static const {c_type} TCN_WEIGHTS_BIASES[] = {{\n")

            for i, value in enumerate(data):
                if i % 8 == 0 and i > 0:
                    f.write("\n    ")
                f.write(f"{value}, ")
            f.write("\n};\n\n")

            f.write(f"#endif // TCN_WEIGHTS_BIASES_{dtype_name.upper()}_H\n")

    @staticmethod
    def export_tcn_weights_combined(model, output_dir, input_data, output_data, scales):
        dtypes = [(np.float32, 'float32'), (np.int32, 'int32'), (np.int16, 'int16'), (np.int8, 'int8')]

        for dtype, dtype_name in dtypes:
            dtype_folder = os.path.join(output_dir, dtype_name)
            os.makedirs(dtype_folder, exist_ok=True)

            combined_weights = []
            combined_biases = []

            for i, layer in enumerate(model.layers):
                # Ottieni pesi e bias dal layer
                w = layer.conv.weight.data.cpu().numpy()
                b = layer.conv.bias.data.cpu().numpy()

                # Quantizza i pesi e bias
                quantized_w = Exporter.quantize_data(w, dtype, scales[dtypes.index((dtype, dtype_name))])
                quantized_b = Exporter.quantize_data(b, dtype, scales[dtypes.index((dtype, dtype_name))])

                # Appendi i pesi e bias all'array combinato
                combined_weights.append(quantized_w.flatten())
                combined_biases.append(quantized_b.flatten())

            # Unisci tutti i pesi e bias in un unico array
            combined_array = np.concatenate(combined_weights + combined_biases)

            # Salva il file header
            header_path = os.path.join(dtype_folder, f"tcn_weights_and_biases_{dtype_name}.h")
            Exporter.generate_combined_header_file(combined_array, header_path, dtype_name)
            Exporter.export_reference_input_output

            # Genera un unico file header per input/output
            ref_header_path = os.path.join(dtype_folder, f"tcn_input_output_{dtype_name}.h")
            quantized_input = Exporter.quantize_data(input_data, dtype, scales[dtypes.index((dtype, dtype_name))])
            quantized_output = Exporter.quantize_data(output_data, dtype, scales[dtypes.index((dtype, dtype_name))])
            Exporter.export_reference_input_output(quantized_input, quantized_output, ref_header_path, dtype_name)

    @staticmethod
    def export_network_parameters(header_path, input_dim, output_dim, hidden_dim, num_layers, kernel_size, sequence_length, fixed_point):
        with open(header_path, 'w') as f:
            f.write("#ifndef TCN_NETWORK_PARAMS_H\n")
            f.write("#define TCN_NETWORK_PARAMS_H\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"#define INPUT_DIM {input_dim}\n")
            f.write(f"#define OUTPUT_DIM {output_dim}\n")
            f.write(f"#define HIDDEN_DIM {hidden_dim}\n")
            f.write(f"#define NUM_LAYERS {num_layers}\n")
            f.write(f"#define KERNEL_SIZE {kernel_size}\n")
            f.write(f"#define TIME_LENGTH {sequence_length}\n")
            f.write(f"#define FIXED_POINT {fixed_point}\n\n")
            f.write("#endif // TCN_NETWORK_PARAMS_H\n")

# --- Main Pipeline ---
file_path = "../NYC_Weather_2016_2022.csv"
input_cols = ["temperature_2m (°C)", "cloudcover (%)", "precipitation (mm)", "winddirection_10m (°)"]
output_cols = ["windspeed_10m (km/h)"]

# Caricamento e pulizia del dataset
data = pd.read_csv(file_path)
data = DataUtils.check_and_clean_data(data, input_cols, output_cols)

dataset = TimeSeriesDataset(data, input_cols, output_cols)
train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modello
input_dim = len(input_cols)
output_dim = len(output_cols)
hidden_dim, kernel_size, num_layers = 16, 4, 4
fixed_point = 12

tcn_model = TCN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)

# Check per modello pre-addestrato
Force_training = True
model_path = "best_model_TCN.pth"
if os.path.exists(model_path) and not Force_training:
    print(f"Caricamento modello pre-addestrato da {model_path}.")
    tcn_model.load_state_dict(torch.load(model_path))
    tcn_model.eval()
else:
    print("> Avvio del training...")
    trainer = Trainer(tcn_model, device, learning_rate=1e-3)
    trained_model, _, _ = trainer.train(train_loader, val_loader, epochs=1000, patience=100)
    torch.save(trained_model.state_dict(), model_path)

# Testing
tester = Tester()
predictions, actuals, inference_time = tester.test_model(tcn_model, test_loader, device)
rmse = tester.calculate_rmse(predictions, actuals)
print(f"\n> RMSE: {rmse:.4f}, Inference Time: {inference_time:.4f} seconds")

plt.figure()
plt.plot(actuals, label="Actual")
plt.plot(predictions, label="TCN Predictions")
plt.xlabel("Time Step")
plt.ylabel("Output")
plt.legend()
plt.title("Predictions TCN")
plt.show()

# Export dei pesi e dei parametri
scales = [1.0, 2**fixed_point, 2**(fixed_point//2), 2**(fixed_point//4)]
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
Exporter.export_tcn_weights(tcn_model, output_dir, scales)

# Ricarichiamo il best model TCN
best_tcn = TCN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers)
best_tcn.load_state_dict(torch.load("best_model_TCN.pth", map_location=device, weights_only=True))
best_tcn.eval()

batch=1
torch_input = torch.randn(batch, input_dim, SEQUENCE_LENGTH, dtype=torch.float32)
with torch.no_grad():
    py_output = best_tcn(torch_input).cpu().numpy() # [batch, output_dim]

output_dir = "../static_ie"
Exporter.export_tcn_weights_combined(tcn_model, output_dir, torch_input.cpu().numpy(), py_output, scales)
Exporter.export_network_parameters(
    header_path=os.path.join(output_dir, "tcn_network_params.h"),
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    kernel_size=kernel_size,
    sequence_length=SEQUENCE_LENGTH,
    fixed_point=fixed_point
)