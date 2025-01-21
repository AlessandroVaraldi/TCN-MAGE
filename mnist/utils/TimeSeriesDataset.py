import torch
import numpy as np
from torch.utils.data import Dataset

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