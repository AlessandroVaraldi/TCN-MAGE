// tcn_network.h
#ifndef TCN_NETWORK_H
#define TCN_NETWORK_H

void tcn_network(float* input, float* output, float** kernels, int input_len, int num_layers, int kernel_size, int stride, int padding);

#endif // TCN_NETWORK_H
