import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = device
        self.max_conv_values = {}  # Dizionario per registrare i massimi valori delle convoluzioni

    def _hook_conv_output(self, module, input, output, layer_name):
        max_value = output.abs().max().item()
        if layer_name not in self.max_conv_values:
            self.max_conv_values[layer_name] = max_value
        else:
            self.max_conv_values[layer_name] = max(self.max_conv_values[layer_name], max_value)

    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv1d):
                layer.register_forward_hook(lambda module, input, output, name=name: self._hook_conv_output(module, input, output, name))

    def train(self, train_loader, val_loader, epochs=1000, patience=50, threshold=1e-4):
        self._register_hooks()  # Registra gli hook per tracciare le convoluzioni
        
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = os.path.join('checkpoints/', f'best_{type(self.model).__name__}.pth')

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
        
        max_values = self._print_max_conv_values()
        
        return self.model, train_losses, val_losses, max_values

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

    def _print_max_conv_values(self):
        max_values = list(self.max_conv_values.values())
        print("\n> Massimi valori delle convoluzioni per ogni layer:")
        for layer_name, max_value in self.max_conv_values.items():
            print(f"  {layer_name}: {max_value:.6f}")
        return max_values
            
