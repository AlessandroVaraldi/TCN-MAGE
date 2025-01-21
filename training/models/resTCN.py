import torch.nn as nn
import torch.nn.functional as F
import json

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
    
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, dilation):
        super(ResidualUnit, self).__init__()
        self.kernel_size = kernel_size
        self.batchnorm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.conv = CausalConv1d(in_channels, hidden_dim, kernel_size, dilation)
        
    def forward(self, x):
        if self.batchnorm:
            out = self.batchnorm(x)
        out = self.relu(out)
        out = self.conv(out)
        return x + out

class ResidualTCN(nn.Module):
    def __init__(self, input_dim, config_path):
        super(ResidualTCN, self).__init__()
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.layers = nn.ModuleList()
        self.hidden_dims = []
        self.kernel_sizes = []
        self.dilations = []

        prev_channels = input_dim

        for layer_config in config['layers']:
            layer_type = layer_config['layer_type']
            hidden_dim = layer_config['hidden_dim']
            kernel_size = layer_config['kernel_size']
            dilation = layer_config['dilation']
            use_gap = layer_config.get('use_gap', False)
            use_softmax = layer_config.get('use_softmax', False)

            layer = nn.Sequential()
            if layer_type == 'residual':
                layer.add_module('residual', ResidualUnit(prev_channels, hidden_dim, kernel_size, dilation))
            else:
                layer.add_module('conv', CausalConv1d(prev_channels, hidden_dim, kernel_size, dilation))
                
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
                            f"  layer_type={'residual' if 'residual' in layer._modules else 'conv'}, "
                            f"  hidden_dim={self.hidden_dims[i]}, "
                            f"  kernel_size={self.kernel_sizes[i]}, "
                            f"  dilation={self.dilations[i]}, "
                            f"  gap={'yes' if 'gap' in layer._modules else 'no'}, "
                            f"  softmax={'yes' if 'softmax' in layer._modules else 'no'}\n")
        return description