#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tcn.h"
#include "utils.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <weights_dir>\n", argv[0]);
        return 1;
    }

    const char *weights_dir = argv[1];

    char INFO_MODE = 0;

    if (argc > 2 && strcmp(argv[2], "info") == 0) {
        printf("Info mode attivato.\n");
        INFO_MODE = 1;
    }

    int input_dim = 7;
    int output_dim = 1;
    int hidden_dim = 8;
    int num_layers = 4;
    int kernel_size = 3;
    int batch = 1;
    int time_length = 256;

    printf("Parametri: input_dim=%d, output_dim=%d, hidden_dim=%d, num_layers=%d, kernel_size=%d, batch=%d, time_length=%d\n\n",
           input_dim, output_dim, hidden_dim, num_layers, kernel_size, batch, time_length);

    // Inizializza modello
    TCNModel *model = tcn_init(input_dim, output_dim, hidden_dim, num_layers-1, kernel_size);

    // Carico pesi
    if (load_tcn_weights(model, weights_dir) < 0) {
        printf("Errore nel caricamento dei pesi.\n");
        tcn_free(model);
        return 1;
    }

    // Carico input salvato da Python
    // Dimensione: batch * input_dim * time_length * sizeof(float)
    size_t input_size = (size_t)batch * (size_t)input_dim * (size_t)time_length;
    float *input_data = (float *)malloc(input_size * sizeof(float));
    FILE *f = fopen("../../input_data.bin", "rb");
    if (!f) {
        printf("Impossibile aprire input_data.bin\n");
        tcn_free(model);
        free(input_data);
        return 1;
    }
    size_t read_count = fread(input_data, sizeof(float), input_size, f);
    fclose(f);
    if (read_count != input_size) {
        printf("Lettura input_data.bin incompleta: letti %zu float, attesi %zu\n", read_count, input_size);
        tcn_free(model);
        free(input_data);
        return 1;
    }

    // Alloca output
    float *output = (float *)malloc(sizeof(float)*batch*output_dim);

    // Forward
    tcn_forward(model, input_data, output, batch, time_length, INFO_MODE);

    // Stampa output ottenuto dal C
    printf("Output dal C: ");
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < output_dim; j++) {
            printf("%.6f ", output[b*output_dim + j]);
        }
        printf("\n");
    }

    // Confronto con output di riferimento Python (opzionale)
    // Leggi il file output_reference.bin e confronta
    FILE *fref = fopen("../../output_reference.bin", "rb");
    if (fref) {
        float ref_out;
        size_t ref_count = fread(&ref_out, sizeof(float), 1, fref);
        fclose(fref);
        if (ref_count == 1) {
            printf("Output di riferimento Python: %.6f\n", ref_out);
            printf("Differenza: %.6f\n", fabs(output[0] - ref_out));
        } else {
            printf("Impossibile leggere il riferimento da output_reference.bin\n");
        }
    }

    // Cleanup
    free(input_data);
    free(output);
    tcn_free(model);

    return 0;
}
