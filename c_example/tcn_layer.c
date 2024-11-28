#include "tcn_layer.h"
#include <math.h>

// Funzione ReLU
float relu(float x) {
    return (x > 0) ? x : 0;
}

// Implementazione della convoluzione con dilatazione
void causal_convolution_1d(
    const float input[L_TILE + K - 1][C_IN], 
    const float kernel[K][C_IN][C_OUT], 
    float output[L_TILE][C_OUT], 
    int dilation_rate) {
    for (int t = 0; t < L_TILE; t++) {
        for (int co = 0; co < C_OUT; co++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                for (int ci = 0; ci < C_IN; ci++) {
                    int idx = t + K - 1 - k * dilation_rate;
                    if (idx >= 0) {
                        sum += input[idx][ci] * kernel[k][ci][co];
                    }
                }
            }
            output[t][co] = sum;
        }
    }
}

// Funzione per applicare un'attivazione
void apply_activation(
    float output[L_TILE][C_OUT], 
    int activation_type) {
    for (int t = 0; t < L_TILE; t++) {
        for (int co = 0; co < C_OUT; co++) {
            if (activation_type == ACTIVATION_RELU) {
                output[t][co] = relu(output[t][co]);
            }
        }
    }
}
