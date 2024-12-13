#ifndef TCN_H
#define TCN_H

#include "causal_conv1d.h"

typedef struct {
    int input_dim;
    int output_dim;
    int hidden_dim;
    int num_layers;
    int kernel_size;
    // array di layer
    CausalConv1dLayer *layers;
    // Funzione di attivazione ReLU applicata a tutti i layer tranne l’ultimo
    // Gestita in tcn_forward.
} TCNModel;

/**
 * Inizializza il modello TCN con i parametri forniti e allocando memoria per i pesi.
 * I pesi andranno caricati con funzioni apposite.
 */
TCNModel *tcn_init(int input_dim, int output_dim, int hidden_dim, int num_layers, int kernel_size);

/**
 * Libera la memoria del modello TCN.
 */
void tcn_free(TCNModel *model);

/**
 * Esegue l'inferenza forward del TCN su un batch di dati.
 * input: [batch, input_dim, time]
 * output: [batch, output_dim], prendendo l'ultimo timestep.
 */
void tcn_forward(TCNModel *model, const float *input, float *output, int batch, int time_length);

#endif
