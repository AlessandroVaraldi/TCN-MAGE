#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tcn_layer.h"

// Numero di layer nel modello
#define NUM_LAYERS 3

// Struttura del modello TCN
typedef struct {
    float kernel[NUM_LAYERS][K][C_IN][C_OUT];
    int dilation_rates[NUM_LAYERS];
} TCNModel;

// Inizializza il modello con pesi casuali e dilatazioni incrementali
void initialize_model(TCNModel *model) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int k = 0; k < K; k++) {
            for (int ci = 0; ci < C_IN; ci++) {
                for (int co = 0; co < C_OUT; co++) {
                    model->kernel[l][k][ci][co] = (float)rand() / RAND_MAX;
                }
            }
        }
        model->dilation_rates[l] = 1 << l; // Dilatazione incrementale: 1, 2, 4, ...
    }
}

// Esegue il modello TCN su un input
void run_tcn_model(
    TCNModel *model, 
    const float input[L_TILE + K - 1][C_IN], 
    float output[L_TILE][C_OUT]) {
    float layer_input[L_TILE + K - 1][C_IN];
    float layer_output[L_TILE][C_OUT];

    // Copia l'input iniziale
    for (int t = 0; t < L_TILE + K - 1; t++) {
        for (int ci = 0; ci < C_IN; ci++) {
            layer_input[t][ci] = input[t][ci];
        }
    }

    // Passa attraverso ogni layer
    for (int l = 0; l < NUM_LAYERS; l++) {
        causal_convolution_1d(layer_input, model->kernel[l], layer_output, model->dilation_rates[l]);
        apply_activation(layer_output, ACTIVATION_RELU);

        // Preparazione per il layer successivo
        for (int t = 0; t < L_TILE; t++) {
            for (int co = 0; co < C_OUT; co++) {
                layer_input[t + K - 1][0] = layer_output[t][co];
            }
        }
    }

    // Copia l'output finale
    for (int t = 0; t < L_TILE; t++) {
        for (int co = 0; co < C_OUT; co++) {
            output[t][co] = layer_output[t][co];
        }
    }
}

int main() {
    TCNModel model;
    float input[L_TILE + K - 1][C_IN] = {0};
    float output[L_TILE][C_OUT] = {0};

    // Inizializza il modello
    initialize_model(&model);

    // Inizializza l'input con un segnale sinusoidale
    for (int t = 0; t < L_TILE + K - 1; t++) {
        for (int ci = 0; ci < C_IN; ci++) {
            input[t][ci] = sinf(2 * M_PI * t / 50.0f);
        }
    }

    // Esegui il modello
    run_tcn_model(&model, input, output);

    // Output di esempio
    printf("Output finale del modello TCN:\n");
    for (int t = 0; t < 10; t++) {
        for (int co = 0; co < C_OUT; co++) {
            printf("%.2f ", output[t][co]);
        }
        printf("\n");
    }

    return 0;
}
