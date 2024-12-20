#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_array(const char *filename, float *data, size_t expected_size) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Impossibile aprire il file %s.\n", filename);
        return -1;
    }
    size_t read_count = fread(data, sizeof(float), expected_size, f);
    fclose(f);

    if (read_count != expected_size) {
        fprintf(stderr, "Errore nella lettura di %s: letti %zu elementi, attesi %zu.\n", filename, read_count, expected_size);
        return -1;
    }
    return 0;
}

int load_tcn_weights(TCNModel *model, const char *weights_dir) {
    // qui costruisco i nomi dei file in modo semplificato
    char path[256];
    for (int i = 0; i < model->num_layers; i++) {
        int in_ch = model->layers[i].in_channels;
        int out_ch = model->layers[i].out_channels;
        int ks = model->layers[i].kernel_size;

        // Struttura attesa per questo layer
        size_t expected_weight_size = (size_t)out_ch * (size_t)in_ch * (size_t)ks;
        //printf("Layer %d: %d x %d x %d = %zu\n", i, out_ch, in_ch, ks, expected_weight_size);
        size_t expected_bias_size = (size_t)out_ch;

        snprintf(path, sizeof(path), "%s/layer%d_weight.bin", weights_dir, i);
        if (load_array(path, model->layers[i].weight, expected_weight_size) < 0) {
            fprintf(stderr, "Errore nel caricamento dei pesi del layer %d.\n", i);
            //fprintf(stderr, "Struttura attesa: weight[%d, %d, %d] = %zu float\n", out_ch, in_ch, ks, expected_weight_size);
            return -1;
        }

        snprintf(path, sizeof(path), "%s/layer%d_bias.bin", weights_dir, i);
        if (load_array(path, model->layers[i].bias, expected_bias_size) < 0) {
            fprintf(stderr, "Errore nel caricamento dei bias del layer %d.\n", i);
            //fprintf(stderr, "Struttura attesa: bias[%d] = %zu float\n", out_ch, expected_bias_size);
            return -1;
        }
    }
    return 0;
}

