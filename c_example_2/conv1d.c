// tcn_layer.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Dichiarazione della funzione di convoluzione
extern void conv1d(float* input, float* output, float* kernel, int input_len, int kernel_size, int stride, int padding);

void relu(float* input, float* output, int len) {
    for (int i = 0; i < len; i++) {
        output[i] = fmax(0.0f, input[i]);  // Funzione ReLU
    }
}

void tcn_layer(float* input, float* output, float* kernel, int input_len, int kernel_size, int stride, int padding) {
    float* conv_output = (float*)malloc(input_len * sizeof(float));
    conv1d(input, conv_output, kernel, input_len, kernel_size, stride, padding);
    relu(conv_output, output, input_len);  // Applicazione di ReLU
    free(conv_output);
}
