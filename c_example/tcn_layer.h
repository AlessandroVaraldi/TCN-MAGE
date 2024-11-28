#ifndef TCN_LAYER_H
#define TCN_LAYER_H

#define L_TILE 1024   // Lunghezza del tile
#define C_IN 1        // Numero di canali di input
#define C_OUT 8       // Numero di canali di output
#define K 64          // Lunghezza del kernel
#define ACTIVATION_RELU 1

// Funzioni del layer
void causal_convolution_1d(
    const float input[L_TILE + K - 1][C_IN], 
    const float kernel[K][C_IN][C_OUT], 
    float output[L_TILE][C_OUT], 
    int dilation_rate);

void apply_activation(
    float output[L_TILE][C_OUT], 
    int activation_type);

#endif

