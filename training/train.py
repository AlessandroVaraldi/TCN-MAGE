import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import time
import os
import json

# Verifica se CUDA è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === Imposta il seed per la riproducibilità ===
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

# --- Dataset Utilities ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_cols, output_col, sequence_length, num_classes=4):
        self.X = torch.tensor(data[input_cols].values, dtype=torch.float32)
        self.y, self.class_ranges = self._dynamic_encode_labels(data[output_col].values, num_classes)
        self.sequence_length = sequence_length

    def _dynamic_encode_labels(self, labels, num_classes):
        """
        Suddivide dinamicamente i valori delle etichette in classi bilanciate basate su quantili.

        :param labels: Array dei valori target
        :param num_classes: Numero di classi desiderate
        :return: (Tensor delle etichette, range delle classi)
        """
        percentiles = np.linspace(0, 100, num_classes + 1)
        bins = np.percentile(labels, percentiles)
        class_ranges = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

        encoded_labels = []
        for label in labels:
            for i, (lower, upper) in enumerate(class_ranges):
                if lower <= label < upper or (i == len(class_ranges) - 1 and label == upper):
                    encoded_labels.append(i)
                    break

        return torch.tensor(encoded_labels, dtype=torch.long), class_ranges

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        x_sequence = self.X[idx:idx + self.sequence_length].T
        y_value = self.y[idx + self.sequence_length - 1]
        return x_sequence, y_value

class DataUtils:
    @staticmethod
    def check_and_clean_data(data, input_cols, output_col):
        nan_count = data[input_cols + [output_col]].isna().sum().sum()
        if nan_count > 0:
            print(f"\n> Avviso: trovati {nan_count} valori NaN nel dataset. Verranno rimossi automaticamente.\n")
            data = data.dropna(subset=input_cols + [output_col]).reset_index(drop=True)
        return data

# --- Model Definitions ---
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_batchnorm=False):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.batchnorm = nn.BatchNorm1d(out_channels) if use_batchnorm else None

    def forward(self, x):
        padding = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (padding, 0))
        out = self.conv(x)
        if self.batchnorm:
            out = self.batchnorm(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_dim, config_path):
        super(TCN, self).__init__()
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.layers = nn.ModuleList()
        self.hidden_dims = []
        self.kernel_sizes = []
        self.dilations = []

        prev_channels = input_dim

        for layer_config in config['layers']:
            hidden_dim = layer_config['hidden_dim']
            kernel_size = layer_config['kernel_size']
            dilation = layer_config['dilation']
            use_batchnorm = layer_config.get('use_batchnorm', False)
            use_relu = layer_config.get('use_relu', True)
            use_maxpool = layer_config.get('use_maxpool', False)
            use_dropout = layer_config.get('use_dropout', False)
            dropout_prob = layer_config.get('dropout_prob', 0.0)
            use_gap = layer_config.get('use_gap', False)
            use_softmax = layer_config.get('use_softmax', False)

            layer = nn.Sequential()
            layer.add_module('conv', CausalConv1d(prev_channels, hidden_dim, kernel_size, dilation, use_batchnorm))
            if use_relu:
                layer.add_module('relu', nn.ReLU())
            if use_maxpool:
                layer.add_module('maxpool', nn.MaxPool1d(kernel_size=2))
            if use_dropout:
                layer.add_module('dropout', nn.Dropout(dropout_prob))
            if use_gap:
                layer.add_module('gap', nn.AdaptiveAvgPool1d(1))
            if use_softmax:
                layer.add_module('softmax', nn.Softmax(dim=1))

            self.layers.append(layer)
            self.hidden_dims.append(hidden_dim)
            self.kernel_sizes.append(kernel_size)
            self.dilations.append(dilation)
            prev_channels = hidden_dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if x.size(1) > 1:
            x = x.squeeze(-1)
        return x
    
    def describe(self):
        description = "> TCN Model Structure:\n"
        for i, layer in enumerate(self.layers):
            description += (f"  Layer {i}: "
                            f"  hidden_dim={self.hidden_dims[i]}, "
                            f"  kernel_size={self.kernel_sizes[i]}, "
                            f"  dilation={self.dilations[i]}, "
                            f"  batchnorm={'yes' if 'batchnorm' in layer._modules['conv']._modules else 'no'}, "
                            f"  relu={'yes' if 'relu' in layer._modules else 'no'}, "
                            f"  maxpool={'yes' if 'maxpool' in layer._modules else 'no'}, "
                            f"  dropout={'yes' if 'dropout' in layer._modules else 'no'}, "
                            f"  dropout_prob={layer._modules['dropout'].p if 'dropout' in layer._modules else 0}, "
                            f"  gap={'yes' if 'gap' in layer._modules else 'no'}, "
                            f"  softmax={'yes' if 'softmax' in layer._modules else 'no'}\n")
        return description

# --- Training Utilities ---
class Trainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device

    def train(self, train_loader, val_loader, epochs=1000, patience=50, threshold=1e-4):
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = f'best_model_{type(self.model).__name__}.pth'
        best_model_path = os.path.join('models/', best_model_path)

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
                    print(f"    Epoch {epoch+1}: Miglioramento della Validation Loss, modello salvato.")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"\n> Early stopping: nessun miglioramento per {patience} epoche consecutive.\n")
                    break
        except KeyboardInterrupt:
            print(f"\n> Training interrotto manualmente alla epoca {epoch}.\n")
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
        ax.annotate(f'Epoch {epoch}', xy=(0.01, 0.01), xycoords='axes fraction', ha='left', va='bottom')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.draw()
        plt.pause(0.01)
        
# --- Export Utilities ---
class Exporter:
    @staticmethod
    def quantize_data(data, dtype, scale):
        """
        Applica la quantizzazione a un array.

        :param data: Array numpy da quantizzare
        :param dtype: Tipo di dato target (es. np.int8, np.float32)
        :param scale: Fattore di scalatura
        :return: Array quantizzato
        """
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
    def fuse_conv_batchnorm(conv_layer, batchnorm):
        """Fonde i parametri di convoluzione e batchnorm."""
        conv = conv_layer.conv 
        if batchnorm is None:
            return conv.weight.data, conv.bias.data

        # Ottieni i parametri di batchnorm
        gamma = batchnorm.weight.data
        beta = batchnorm.bias.data
        mean = batchnorm.running_mean
        var = batchnorm.running_var
        eps = batchnorm.eps

        # Fonde i pesi
        scale = gamma / torch.sqrt(var + eps)
        fused_weight = conv.weight.data * scale[:, None, None]
        if conv.bias is not None:
            fused_bias = (conv.bias.data - mean) * scale + beta
        else:
            fused_bias = (-mean) * scale + beta

        return fused_weight, fused_bias

    @staticmethod
    def export_weights_and_biases(model, header_dir, data_dir, scales):
        """
        Esporta i pesi e i bias separati per layer del modello in array separati e genera gli header invariati.

        :param model: Modello PyTorch
        :param header_dir: Directory per il file header
        :param data_dir: Directory per i file binari/numpy
        :param scales: Lista di fattori di scalatura
        """
        dtypes = [(np.float32, 'float32'), (np.int32, 'int32'), (np.int16, 'int16'), (np.int8, 'int8')]

        for dtype, dtype_name in dtypes:
            scale = scales[dtypes.index((dtype, dtype_name))]
            dtype_folder = os.path.join(data_dir, dtype_name)
            os.makedirs(dtype_folder, exist_ok=True)

            all_weights = []
            all_biases = []

            for layer_idx, layer in enumerate(model.layers):
                causal_conv = layer[0]  # Accede al primo modulo (CausalConv1d) in nn.Sequential
                if isinstance(causal_conv, CausalConv1d):
                    fused_weight, fused_bias = Exporter.fuse_conv_batchnorm(causal_conv, causal_conv.batchnorm)
                    quantized_weights = Exporter.quantize_data(fused_weight.cpu().numpy(), dtype, scale)
                    quantized_biases = Exporter.quantize_data(fused_bias.cpu().numpy(), dtype, scale)

                    # Salva i pesi e bias separati per ogni layer
                    np.save(os.path.join(dtype_folder, f"layer_{layer_idx}_weights.npy"), quantized_weights)
                    np.save(os.path.join(dtype_folder, f"layer_{layer_idx}_biases.npy"), quantized_biases)

                    with open(os.path.join(dtype_folder, f"layer_{layer_idx}_weights.bin"), "wb") as f:
                        f.write(quantized_weights.tobytes())
                    with open(os.path.join(dtype_folder, f"layer_{layer_idx}_biases.bin"), "wb") as f:
                        f.write(quantized_biases.tobytes())

                    # Accumula per concatenare
                    all_weights.append(quantized_weights.flatten())
                    all_biases.append(quantized_biases.flatten())

            # Concatenazione di tutti i pesi e bias per header
            concatenated_weights = np.concatenate(all_weights)
            concatenated_biases = np.concatenate(all_biases)
            concatenated_array = np.concatenate([concatenated_weights, concatenated_biases])

            # Salvataggio in formato header
            header_path = os.path.join(header_dir, dtype_name)
            os.makedirs(header_path, exist_ok=True)
            header_path = os.path.join(header_path, f"weights_bias_{dtype_name}.h")
            with open(header_path, "w") as f:
                f.write(f"#ifndef WEIGHTS_BIAS_{dtype_name.upper()}_H\n")
                f.write(f"#define WEIGHTS_BIAS_{dtype_name.upper()}_H\n\n")
                f.write("#include <stdint.h>\n\n")

                c_type = {
                    'float32': 'float',
                    'int32': 'int32_t',
                    'int16': 'int16_t',
                    'int8': 'int8_t'
                }[dtype_name]

                f.write(f"static const {c_type} WEIGHTS_BIAS[] = {{\n")
                for i, value in enumerate(concatenated_array):
                    if i % 8 == 0 and i > 0:
                        f.write("\n    ")
                    f.write(f"{value}")
                    if i < len(concatenated_array) - 1:
                        f.write(", ")
                f.write("\n};\n\n")

                f.write(f"#endif // WEIGHTS_BIAS_{dtype_name.upper()}_H\n")
                
    @staticmethod
    def export_inputs_outputs(input_data, output_data, header_dir, data_dir, scales):
        """
        Esporta input e output in formato .h, .npy e .bin per tutte le scale.

        :param input_data: Array numpy degli input
        :param output_data: Array numpy degli output
        :param output_dir: Directory di output
        :param scales: Lista di fattori di scalatura
        """
        dtypes = [(np.float32, 'float32'), (np.int32, 'int32'), (np.int16, 'int16'), (np.int8, 'int8')]

        for dtype, dtype_name in dtypes:
            dtype_folder = os.path.join(data_dir, dtype_name)
            os.makedirs(dtype_folder, exist_ok=True)
            scale = scales[dtypes.index((dtype, dtype_name))]

            # Quantizzazione
            quantized_input = Exporter.quantize_data(input_data, dtype, scale)
            quantized_output = Exporter.quantize_data(output_data, dtype, scale)

            # Salvataggio in formato .npy
            np.save(os.path.join(dtype_folder, "input.npy"), quantized_input)
            np.save(os.path.join(dtype_folder, "output.npy"), quantized_output)

            # Salvataggio in formato .bin
            with open(os.path.join(dtype_folder, "input.bin"), "wb") as f:
                f.write(quantized_input.tobytes())
            with open(os.path.join(dtype_folder, "output.bin"), "wb") as f:
                f.write(quantized_output.tobytes())

            # Salvataggio in formato header
            header_path = os.path.join(header_dir, dtype_name)
            os.makedirs(header_path, exist_ok=True)
            header_path = os.path.join(header_path, f"input_output_{dtype_name}.h")
            with open(header_path, "w") as f:
                f.write(f"#ifndef INPUT_OUTPUT_{dtype_name.upper()}_H\n")
                f.write(f"#define INPUT_OUTPUT_{dtype_name.upper()}_H\n\n")
                f.write("#include <stdint.h>\n\n")

                c_type = {
                    'float32': 'float',
                    'int32': 'int32_t',
                    'int16': 'int16_t',
                    'int8': 'int8_t'
                }[dtype_name]

                # Esportazione degli input
                f.write(f"static const {c_type} REF_INPUT[] = {{\n")
                for i, value in enumerate(quantized_input.flatten()):
                    if i % 8 == 0 and i > 0:
                        f.write("\n    ")
                    f.write(f"{value}")
                    if i < len(quantized_input.flatten()) - 1:
                        f.write(", ")
                f.write("\n};\n\n")

                # Esportazione degli output
                f.write(f"static const {c_type} REF_OUTPUT[] = {{\n")
                for i, value in enumerate(quantized_output.flatten()):
                    if i % 8 == 0 and i > 0:
                        f.write("\n    ")
                    f.write(f"{value}")
                    if i < len(quantized_output.flatten()) - 1:
                        f.write(", ")
                f.write("\n};\n\n")

                f.write(f"#endif // INPUT_OUTPUT_{dtype_name.upper()}_H\n")

    @staticmethod
    def export_network_parameters(header_path, model, input_dim, hidden_dims, kernel_sizes, dilations, sequence_length, fixed_point):
        """Esporta i parametri del modello in formato header con array, includendo la presenza di operazioni nei layer."""
        with open(header_path, 'w') as f:
            f.write("#ifndef TCN_NETWORK_PARAMS_H\n")
            f.write("#define TCN_NETWORK_PARAMS_H\n\n")
            f.write("#include <stdint.h>\n\n")

            f.write(f"#define NUM_LAYERS {len(hidden_dims)}\n")
            f.write(f"#define INPUT_DIM {input_dim}\n")
            f.write(f"#define TIME_LENGTH {sequence_length}\n")
            f.write(f"#define FIXED_POINT {fixed_point}\n")
            f.write(f"#define MAX_HIDDEN_DIM {max(hidden_dims)}\n\n")
            f.write(f"#define NUM_CLASSES {hidden_dims[-1]}\n\n")

            f.write(f"static const int32_t HIDDEN_DIMS[] = {{ {', '.join(map(str, hidden_dims))} }};\n")
            f.write(f"static const int32_t KERNEL_SIZES[] = {{ {', '.join(map(str, kernel_sizes))} }};\n")
            f.write(f"static const int32_t DILATIONS[] = {{ {', '.join(map(str, dilations))} }};\n")

            # Genera vettori per operazioni aggiuntive
            relu_flags = [1 if 'relu' in layer._modules else 0 for layer in model.layers]
            maxpool_flags = [1 if 'maxpool' in layer._modules else 0 for layer in model.layers]
            dropout_flags = [1 if 'dropout' in layer._modules else 0 for layer in model.layers]
            gap_flags = [1 if 'gap' in layer._modules else 0 for layer in model.layers]
            softmax_flags = [1 if 'softmax' in layer._modules else 0 for layer in model.layers]

            f.write(f"static const int32_t RELU_FLAGS[] = {{ {', '.join(map(str, relu_flags))} }};\n")
            f.write(f"static const int32_t MAXPOOL_FLAGS[] = {{ {', '.join(map(str, maxpool_flags))} }};\n")
            f.write(f"static const int32_t DROPOUT_FLAGS[] = {{ {', '.join(map(str, dropout_flags))} }};\n")
            f.write(f"static const int32_t GAP_FLAGS[] = {{ {', '.join(map(str, gap_flags))} }};\n")
            f.write(f"static const int32_t SOFTMAX_FLAGS[] = {{ {', '.join(map(str, softmax_flags))} }};\n")

            f.write("\n#endif // TCN_NETWORK_PARAMS_H\n")

# --- Main Pipeline ---
file_path = "../NYC_Weather_2016_2022.csv"
input_cols = ["cloudcover (%)", "precipitation (mm)", "winddirection_10m (°)", "windspeed_10m (km/h)"]
output_col = "temperature_2m (°C)"

# Caricamento e pulizia del dataset
data = pd.read_csv(file_path)
data = DataUtils.check_and_clean_data(data, input_cols, output_col)

# === Sequence Length ===
config_path = "cfg/config1.json"
with open(config_path, 'r') as f:
    config = json.load(f)
sequence_length = config['sequence_length']

# Dataset e DataLoader
num_classes = 4  # Può essere modificato dinamicamente
dataset = TimeSeriesDataset(data, input_cols, output_col, sequence_length=sequence_length, num_classes=num_classes)
train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modello
input_dim = len(input_cols)
tcn_model = TCN(input_dim=input_dim, config_path=config_path).to(device)

# Mostra la struttura del modello
print(tcn_model.describe())

# Check per modello pre-addestrato
Force_training = True
Resume_training = True

if Resume_training:
    model_path = f'models/best_model_{type(tcn_model).__name__}.pth'
    tcn_model.load_state_dict(torch.load(model_path, map_location=device))
    tcn_model.train()
    trainer = Trainer(tcn_model, device, learning_rate=1e-3)
    trained_model, _, _ = trainer.train(train_loader, val_loader, epochs=1000, patience=100)
    torch.save(trained_model.state_dict(), model_path)
elif Force_training:
    model_path = f'models/best_model_{type(tcn_model).__name__}.pth'
    if os.path.exists(model_path):
        os.remove(model_path)
    tcn_model.train()
    trainer = Trainer(tcn_model, device, learning_rate=1e-3)
    trained_model, _, _ = trainer.train(train_loader, val_loader, epochs=1000, patience=100)
    torch.save(trained_model.state_dict(), model_path)
else:
    model_path = f'models/best_model_{type(tcn_model).__name__}.pth'
    if os.path.exists(model_path):
        tcn_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        tcn_model.train()
        trainer = Trainer(tcn_model, device, learning_rate=1e-3)
        trained_model, _, _ = trainer.train(train_loader, val_loader, epochs=10000, patience=1000)
        torch.save(trained_model.state_dict(), model_path)
    

# Testing
tcn_model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = tcn_model(inputs)
        predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())
        actuals.append(targets.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)
accuracy = np.mean(predictions == actuals)
print(f"\n> Accuracy: {accuracy:.4f}")

# Confusion Matrix and Classification Report
cm = confusion_matrix(actuals, predictions)
cmd = ConfusionMatrixDisplay(cm, display_labels=[f"Class {i}: {lower:.2f} to {upper:.2f}" for i, (lower, upper) in enumerate(dataset.class_ranges)])
cmd.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(actuals, predictions, target_names=[f"Class {i}: {lower:.2f} to {upper:.2f}" for i, (lower, upper) in enumerate(dataset.class_ranges)]))

# Ricarichiamo il best model TCN
best_tcn = TCN(input_dim=input_dim, config_path=config_path)
best_tcn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
best_tcn.eval()

batch=1
torch_input = torch.randn(batch, input_dim, sequence_length, dtype=torch.float32)
with torch.no_grad():
    py_output = best_tcn(torch_input).cpu().numpy()
    
# Export dei pesi e dei parametri
fixed_point = 12
scales = [1.0, 2**fixed_point, 2**(fixed_point//2), 2**(fixed_point//4)]

header_dir = "../static_ie"
data_dir = "data"
# Remove existing files
if os.path.exists(data_dir):
    os.system(f"rm -r {data_dir}")
os.makedirs(data_dir, exist_ok=True)
Exporter.export_weights_and_biases(best_tcn, header_dir, data_dir, scales)
Exporter.export_inputs_outputs(torch_input.cpu().numpy(), py_output, header_dir, data_dir, scales)
Exporter.export_network_parameters(
    header_path=os.path.join(header_dir, "tcn_network_params.h"),
    model=best_tcn,
    input_dim=input_dim,
    hidden_dims=tcn_model.hidden_dims,
    kernel_sizes=tcn_model.kernel_sizes,
    dilations=tcn_model.dilations,
    sequence_length=sequence_length,
    fixed_point=fixed_point
)

