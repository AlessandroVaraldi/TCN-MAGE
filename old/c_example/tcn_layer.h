// File: tcn_layer.h

#ifndef TCN_LAYER_H
#define TCN_LAYER_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

// Struttura per un singolo layer TCN
typedef struct {
    int c_in;       // Numero di canali di ingresso
    int c_out;      // Numero di canali di uscita
    int kernel_size; // Dimensione del kernel
    int dilation;   // Distanza tra gli elementi del kernel
    double* weights; // Pesi del layer (matrice c_out x (c_in * kernel_size))
    double* biases;  // Bias per ogni canale di uscita
} TCNLayer;

// Funzione per inizializzare un layer TCN
void initialize_tcn_layer(TCNLayer* layer, int c_in, int c_out, int kernel_size, int dilation);

// Funzione per applicare il layer TCN a una sequenza di dati
void apply_tcn_layer(TCNLayer* layer, double* input_sequence, double* output_sequence, int sequence_length);

#endif // TCN_LAYER_H


