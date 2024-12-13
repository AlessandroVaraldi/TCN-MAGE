// File: tcn_network.h

#ifndef TCN_NETWORK_H
#define TCN_NETWORK_H

#include "tcn_layer.h"

typedef struct {
    int num_layers;        // Numero di layer nella rete
    TCNLayer* layers;      // Array di layer
    int input_channels;    // Numero di canali di ingresso del primo layer
    int output_channels;   // Numero di canali di uscita dell'ultimo layer
    int sequence_length;   // Lunghezza della sequenza di input/output
} TCNNetwork;

// Inizializza la rete TCN
void initialize_tcn_network(TCNNetwork* network, int num_layers, int input_channels, int output_channels, int sequence_length);

// Esegue la rete TCN su una sequenza di input
void apply_tcn_network(TCNNetwork* network, double* input_sequence, double* output_sequence);

// Libera le risorse allocate dalla rete TCN
void free_tcn_network(TCNNetwork* network);

#endif // TCN_NETWORK_H
