// File: tcn_layer.c

#include <stdio.h>
#include "tcn_layer.h"

// Funzione ausiliaria per inizializzare pesi e bias
static void initialize_weights_and_biases(double* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Valori casuali tra -1 e 1
    }
}

// Inizializzazione di un layer TCN
void initialize_tcn_layer(TCNLayer* layer, int c_in, int c_out, int kernel_size, int dilation) {
    layer->c_in = c_in;
    layer->c_out = c_out;
    layer->kernel_size = kernel_size;
    layer->dilation = dilation;
    int weights_size = c_out * c_in * kernel_size;
    layer->weights = (double*)malloc(weights_size * sizeof(double));
    layer->biases = (double*)malloc(c_out * sizeof(double));

    if (layer->weights == NULL || layer->biases == NULL) {
        // Gestione dell'errore di allocazione della memoria
        fprintf(stderr, "Errore di allocazione della memoria nel layer TCN\n");
        exit(1);
    }

    initialize_weights_and_biases(layer->weights, weights_size);
    initialize_weights_and_biases(layer->biases, c_out);
}

// Applicazione di un layer TCN
void apply_tcn_layer(TCNLayer* layer, double* input_sequence, double* output_sequence, int sequence_length) {
    int padding_size = (layer->kernel_size - 1) * layer->dilation;
    int padded_sequence_length = sequence_length + padding_size;
    
    // Alloca memoria per la sequenza di input con padding
    double* padded_input = (double*)calloc(padded_sequence_length * layer->c_in, sizeof(double));
    if (padded_input == NULL) {
        fprintf(stderr, "Failed to allocate memory for padded input sequence\n");
        exit(1);
    }

    // Copia l'input originale nella nuova sequenza, spostando in avanti per il padding
    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < layer->c_in; j++) {
            padded_input[(i + padding_size) * layer->c_in + j] = input_sequence[i * layer->c_in + j];
        }
    }

    // Applica la convoluzione usando la sequenza con padding
    memset(output_sequence, 0, sequence_length * layer->c_out * sizeof(double));
    for (int i = 0; i < sequence_length; i++) {
        for (int co = 0; co < layer->c_out; co++) {
            double output = 0.0;
            for (int ki = 0; ki < layer->kernel_size; ki++) {
                int input_index = i + ki * layer->dilation;  // Cambiato l'offset di input_index per considerare il padding
                for (int ci = 0; ci < layer->c_in; ci++) {
                    int weight_index = co * (layer->c_in * layer->kernel_size) + ki * layer->c_in + ci;
                    output += layer->weights[weight_index] * padded_input[input_index * layer->c_in + ci];
                }
            }
            output += layer->biases[co];
            output_sequence[i * layer->c_out + co] = output;
        }
    }

    // Libera la memoria allocata per il padded input
    free(padded_input);
}