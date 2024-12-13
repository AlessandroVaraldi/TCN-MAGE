#include <stdio.h>
#include <stdlib.h>
#include "tcn.h"
#include "utils.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <weights_dir>\n", argv[0]);
        return 1;
    }

    const char *weights_dir = argv[1];

    int input_dim = 7;
    int output_dim = 1;
    int hidden_dim = 16;
    int num_layers = 3;
    int kernel_size = 3;
    int batch = 1;
    int time_length = 50;

    // Inizializza modello
    TCNModel *model = tcn_init(input_dim, output_dim, hidden_dim, num_layers, kernel_size);

    // Carico pesi
    if (load_tcn_weights(model, weights_dir) < 0) {
        printf("Errore nel caricamento dei pesi.\n");
        tcn_free(model);
        return 1;
    }

    // Preparo input dummy: [batch=1, input_dim=1, time_length=50]
    float *input = (float *)malloc(sizeof(float)*batch*input_dim*time_length);
    for (int i = 0; i < batch*input_dim*time_length; i++) {
        input[i] = (float)(i % 10); // un pattern semplice
    }

    // Output: [batch, output_dim]
    float *output = (float *)malloc(sizeof(float)*batch*output_dim);

    // Eseguo forward
    tcn_forward(model, input, output, batch, time_length);

    // Stampa output
    printf("Output finale:\n");
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < output_dim; j++) {
            printf("%.4f ", output[b*output_dim + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(input);
    free(output);
    tcn_free(model);

    return 0;
}
