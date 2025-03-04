import os
import json
import torch
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from models.tcn import TCN, CausalConv1d
from utils.TimeSeriesDataset import TimeSeriesDataset
from utils.Trainer import Trainer

# Verifica se CUDA è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === Imposta il seed per la riproducibilità ===
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

class DataUtils:
    @staticmethod
    def check_and_clean_data(data, input_cols, output_col):
        nan_count = data[input_cols + [output_col]].isna().sum().sum()
        if nan_count > 0:
            print(f"\n> Avviso: trovati {nan_count} valori NaN nel dataset. Verranno rimossi automaticamente.\n")
            data = data.dropna(subset=input_cols + [output_col]).reset_index(drop=True)
        return data
        
# --- Export Utilities ---
class Exporter:
    @staticmethod
    def quantize_data(data, dtype, scale):
        """
        Applies quantization to an array.

        :param data: Numpy array to quantize
        :param dtype: Target data type (e.g., np.int8, np.float32)
        :param scale: Scaling factor
        :return: Quantized array
        """
        scaled_data = data * scale
        scaled_data = np.round(scaled_data) if dtype != np.float32 else scaled_data
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
    def fuse_batchnorm_network(model):
        """
        Fonde la batch normalization in tutta la rete, modificando il modello in-place.

        :param model: Modello TCN PyTorch con potenziali layer CausalConv1d + BatchNorm1d.
        :return: Modello con BatchNorm fuso nei layer convolutivi.
        """
        for layer_idx, layer in enumerate(model.layers):
            # Accede al primo modulo nel layer (dovrebbe essere CausalConv1d)
            causal_conv = layer[0]  # In nn.Sequential, il primo modulo è la convoluzione
            if isinstance(causal_conv, CausalConv1d) and causal_conv.batchnorm is not None:
                print(f"Fusione della BatchNorm per il layer {layer_idx}...")

                conv = causal_conv.conv
                batchnorm = causal_conv.batchnorm

                # Ottieni i parametri di BatchNorm
                gamma = batchnorm.weight.data
                beta = batchnorm.bias.data
                mean = batchnorm.running_mean
                var = batchnorm.running_var
                eps = batchnorm.eps

                # Calcola il fattore di scala e adatta i pesi e i bias
                scale = gamma / torch.sqrt(var + eps)
                conv.weight.data.mul_(scale[:, None, None])  # Modifica in-place i pesi

                if conv.bias is not None:
                    conv.bias.data.sub_(mean).mul_(scale).add_(beta)
                else:
                    conv.bias = torch.nn.Parameter((-mean) * scale + beta)

                # Rimuove il layer di BatchNorm dopo la fusione
                causal_conv.batchnorm = None

        print("Fusione BatchNorm completata su tutta la rete.")
        return model

    @staticmethod
    def fuse_conv_batchnorm(conv_layer, batchnorm):
        """
        Fuses the parameters of convolution and batchnorm layers.

        :param conv_layer: Convolution layer
        :param batchnorm: Batch normalization layer
        :return: Fused weights and biases
        """
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
        conv.weight.data.mul_(scale[:, None, None])
        if conv.bias is not None:
            conv.bias.data.sub_(mean).mul_(scale).add_(beta)
        else:
            conv.bias = torch.nn.Parameter((-mean) * scale + beta)

        return conv.weight.data, conv.bias.data

    @staticmethod
    def export_weights_and_biases(model, header_dir, data_dir, scales):
        """
        Esporta i pesi e i bias separati per layer del modello in array separati e genera gli header invariati.

        :param model: Modello PyTorch
        :param header_dir: Directory per il file header
        :param data_dir: Directory per i file binari/numpy
        :param scales: Lista di fattori di scalatura
        """
        # Fondere BatchNorm prima dell'esportazione
        model = Exporter.fuse_batchnorm_network(model)

        dtypes = [(np.float32, 'float32'), (np.int32, 'int32'), (np.int16, 'int16'), (np.int8, 'int8')]

        for dtype, dtype_name in dtypes:
            scale = scales[dtypes.index((dtype, dtype_name))]
            dtype_folder = os.path.join(data_dir, dtype_name)
            os.makedirs(dtype_folder, exist_ok=True)

            all_weights = []
            all_biases = []

            for layer_idx, layer in enumerate(model.layers):
                causal_conv = layer[0]  # Accede al primo modulo (CausalConv1d) in nn.Sequential
                fused_weight = causal_conv.conv.weight.data.cpu().numpy()
                
                # Riordina i canali di input: per ogni output channel,
                # metti prima tutti i canali pari, poi quelli dispari
                num_in_channels = fused_weight.shape[1]
                even_idx = np.arange(0, num_in_channels, 2)
                odd_idx = np.arange(1, num_in_channels, 2)
                reordered_weight = np.concatenate([fused_weight[:, even_idx, :], fused_weight[:, odd_idx, :]], axis=1)
                
                fused_bias = causal_conv.conv.bias.data.cpu().numpy()

                quantized_weights = Exporter.quantize_data(reordered_weight, dtype, scale)
                quantized_biases = Exporter.quantize_data(fused_bias, dtype, scale)

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

            # Salvataggio in formato header
            header_path = os.path.join(header_dir, f"weights_bias_{dtype_name}.h")
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
                k = 0
                f.write(f"static const {c_type} __attribute__((section(\".xheep_data_flash_only\"))) WEIGHTS[] = {{\n")
                ("\n    ")
                for i, value in enumerate(concatenated_weights):
                    if i % 8 == 0 and i > 0:
                        f.write("\n    ")
                    f.write(f"{value}")
                    if i < len(concatenated_weights) - 1:
                        f.write(", ")
                f.write("\n};\n\n")

                f.write(f"static const {c_type} __attribute__((section(\".xheep_data_flash_only\"))) BIASES[] = {{\n")
                for i, value in enumerate(concatenated_biases):
                    if i % 8 == 0 and i > 0:
                        f.write("\n    ")
                    f.write(f"{value}")
                    if i < len(concatenated_biases) - 1:
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
            
            # Riordina gli input: per ogni time step, metti i canali pari prima e quelli dispari dopo.
            # Assumiamo input_data di forma (batch, channels, time_steps)
            num_channels = input_data.shape[1]
            even_idx = np.arange(0, num_channels, 2)
            odd_idx = np.arange(1, num_channels, 2)
            reordered_input_data = np.concatenate([input_data[:, even_idx, :], input_data[:, odd_idx, :]], axis=1)

            # Quantizzazione
            quantized_input = Exporter.quantize_data(reordered_input_data, dtype, scale)
            quantized_output = Exporter.quantize_data(output_data, dtype, scale)

            # Salvataggio in formato .npy per batch
            np.save(os.path.join(dtype_folder, f"input_{batch}.npy"), quantized_input)
            np.save(os.path.join(dtype_folder, f"output_{batch}.npy"), quantized_output)

            # Salvataggio in formato .bin
            with open(os.path.join(dtype_folder, f"input_{batch}.bin"), "wb") as f:
                f.write(quantized_input.tobytes())
            with open(os.path.join(dtype_folder, f"output_{batch}.bin"), "wb") as f:
                f.write(quantized_output.tobytes())
                
        dtype, dtype_name = dtypes[0]

        # Salvataggio in formato header
        header_path = os.path.join(header_dir, f"input_output.h")
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
            for i, value in enumerate(input_data.flatten()):
                if i % 8 == 0 and i > 0:
                    f.write("\n    ")
                f.write(f"{value}")
                if i < len(input_data.flatten()) - 1:
                    f.write(", ")
            f.write("\n};\n\n")

            # Esportazione degli output
            f.write(f"static const {c_type} REF_OUTPUT[] = {{\n")
            for i, value in enumerate(output_data.flatten()):
                if i % 8 == 0 and i > 0:
                    f.write("\n    ")
                f.write(f"{value}")
                if i < len(output_data.flatten()) - 1:
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
            avgpool_flags = [1 if 'avgpool' in layer._modules else 0 for layer in model.layers]
            gap_flags = [1 if 'gap' in layer._modules else 0 for layer in model.layers]
            softmax_flags = [1 if 'softmax' in layer._modules else 0 for layer in model.layers]

            f.write(f"static const int32_t RELU_FLAGS[] = {{ {', '.join(map(str, relu_flags))} }};\n")
            f.write(f"static const int32_t MAXPOOL_FLAGS[] = {{ {', '.join(map(str, maxpool_flags))} }};\n")
            f.write(f"static const int32_t AVGPOOL_FLAGS[] = {{ {', '.join(map(str, avgpool_flags))} }};\n")
            f.write(f"static const int32_t GAP_FLAGS[] = {{ {', '.join(map(str, gap_flags))} }};\n")
            f.write(f"static const int32_t SOFTMAX_FLAGS[] = {{ {', '.join(map(str, softmax_flags))} }};\n")

            f.write("\n#endif // TCN_NETWORK_PARAMS_H\n")

# --- Main Pipeline ---
file_path = "NYC_Weather_2016_2022.csv"
input_cols = ["cloudcover (%)", "precipitation (mm)", "winddirection_10m (°)", "windspeed_10m (km/h)"]
output_col = "temperature_2m (°C)"

# Caricamento e pulizia del dataset
data = pd.read_csv(file_path)
data = DataUtils.check_and_clean_data(data, input_cols, output_col)

# === Sequence Length ===
config_path = "training/cfg/config3.json"
with open(config_path, 'r') as f:
    config = json.load(f)
sequence_length = config['sequence_length']

# Dataset e DataLoader
num_classes = 4
dataset = TimeSeriesDataset(data, input_cols, output_col, sequence_length=sequence_length, num_classes=num_classes)
train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modello
input_dim = len(input_cols)
tcn_model = TCN(input_dim=input_dim, config_path=config_path).to(device)

# Mostra la struttura del modello
print(tcn_model.describe())

# Check per modello pre-addestrato
model_path = f'training/checkpoints/best_{type(tcn_model).__name__}.pth'

epochs = 10000
patience = 1000

force_training = False
resume_training = False

def train_and_save_model():
    tcn_model.train()
    trainer = Trainer(tcn_model, device, learning_rate=1e-3)
    trained_model, _, _ = trainer.train(train_loader, val_loader, epochs=epochs, patience=patience)
    torch.save(trained_model.state_dict(), model_path)
    
if resume_training:
    if os.path.exists(model_path):
        tcn_model.load_state_dict(torch.load(model_path, map_location=device))
    train_and_save_model()
elif force_training:
    if os.path.exists(model_path):
        os.remove(model_path)
    train_and_save_model()
else:
    if os.path.exists(model_path):
        tcn_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        train_and_save_model()
    
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

header_dir = "even_odd"
data_dir = "training/even_odd"
# Remove existing files
if os.path.exists(data_dir):
    os.system(f"rm -r {data_dir}")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(header_dir, exist_ok=True)
Exporter.export_weights_and_biases(best_tcn, header_dir, data_dir, scales)

torch_input = torch.randn(batch, input_dim, sequence_length, dtype=torch.float32)
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
