## Script to load bin file and display the data

import sys
import numpy as np


def load_bin_file(file_name):
    with open(file_name, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data

file_name = sys.argv[1]
data = load_bin_file(file_name)
print("Data loaded from file: ", file_name)
print("Data shape: ", data.shape)
print(data)