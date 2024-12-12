import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, layers_config, sequence_length):
        """
        layers_config: lista di dizionari che definisce i parametri di ciascun layer.
        Ogni dizionario deve contenere:
        - 'input_channels': numero di canali di ingresso per il layer
        - 'output_channels': numero di canali di uscita per il layer
        - 'kernel_size': dimensione del kernel (opzionale, default 3)
        - 'dilation': fattore di dilatazione per il layer
        """
        super(TCN, self).__init__()
        
        layers = []
        for layer_config in layers_config:
            layers.append(self._create_tcn_block(layer_config))
        
        self.network = nn.Sequential(*layers)
    
    def _create_tcn_block(self, layer_config):
        input_channels = layer_config['input_channels']
        output_channels = layer_config['output_channels']
        kernel_size = layer_config.get('kernel_size', 3)  # Imposta a 3 se non specificato
        dilation = layer_config['dilation']
        
        # Strato convoluzionale 1D con padding per la dilatazione
        return nn.Conv1d(input_channels, output_channels, kernel_size, 
                         padding=(kernel_size - 1) * dilation, dilation=dilation)
    
    def forward(self, x):
        return self.network(x)
