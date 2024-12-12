// tcn_network.c
#include <stdio.h>
#include <stdlib.h>

extern void tcn_layer(float* input, float* output, float* kernel, int input_len, int kernel_size, int stride, int padding);

// Funzione per eseguire l'inferenza nel network TCN
void tcn_network(float* input, float* output, float** kernels, int input_len, int num_layers, int kernel_size, int stride, int padding) {
    float* layer_input = input;
    float* layer_output = (float*)malloc(input_len * sizeof(float));

    for (int i = 0; i < num_layers - 1; i++) {
        tcn_layer(layer_input, layer_output, kernels[i], input_len, kernel_size, stride, padding);
        layer_input = layer_output;  // Passa l'output al prossimo layer
    }

    // Output finale (senza ReLU nel layer finale)
    tcn_layer(layer_input, output, kernels[num_layers - 1], input_len, kernel_size, stride, padding);

    free(layer_output);
}
