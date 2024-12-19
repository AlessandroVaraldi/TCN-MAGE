#include "tcn.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

TCNModel *tcn_init(int input_dim, int output_dim, int hidden_dim, int num_layers, int kernel_size) {
    TCNModel *model = (TCNModel *)malloc(sizeof(TCNModel));
    model->input_dim = input_dim;
    model->output_dim = output_dim;
    model->hidden_dim = hidden_dim;
    model->num_layers = num_layers + 1; // Aggiunto un layer finale senza dilation
    model->kernel_size = kernel_size;

    model->layers = (CausalConv1dLayer *)malloc(sizeof(CausalConv1dLayer)*model->num_layers);

    // Primo layer: input_dim -> hidden_dim
    model->layers[0].in_channels = input_dim;
    model->layers[0].out_channels = hidden_dim;
    model->layers[0].kernel_size = kernel_size;
    model->layers[0].dilation = 1;
    model->layers[0].weight = (float *)malloc(sizeof(float)*hidden_dim*input_dim*kernel_size);
    model->layers[0].bias = (float *)malloc(sizeof(float)*hidden_dim);

    // Layer intermedi: hidden_dim -> hidden_dim con dilazioni
    for (int i = 0; i < num_layers-1; i++) {
        int dilation = 2 << i; // 2^i
        model->layers[i+1].in_channels = hidden_dim;
        model->layers[i+1].out_channels = hidden_dim;
        model->layers[i+1].kernel_size = kernel_size;
        model->layers[i+1].dilation = dilation;
        model->layers[i+1].weight = (float *)malloc(sizeof(float)*hidden_dim*hidden_dim*kernel_size);
        model->layers[i+1].bias = (float *)malloc(sizeof(float)*hidden_dim);
    }

    // Ultimo layer con dilation
    {
        int i = num_layers;
        int dilation = 2 << (num_layers - 1);
        model->layers[i].in_channels = hidden_dim;
        model->layers[i].out_channels = output_dim;
        model->layers[i].kernel_size = kernel_size;
        model->layers[i].dilation = dilation;
        model->layers[i].weight = (float *)malloc(sizeof(float)*hidden_dim*hidden_dim*kernel_size);
        model->layers[i].bias = (float *)malloc(sizeof(float)*hidden_dim);
    }

    return model;
}

void tcn_free(TCNModel *model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; i++) {
        free(model->layers[i].weight);
        free(model->layers[i].bias);
    }
    free(model->layers);
    free(model);
}

static void relu_inplace(float *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void tcn_forward(TCNModel *model, const float *input, float *output, int batch, int time_length, char INFO_MODE) {
    // input: [batch, input_dim, time_length]
    // dimensione temporanea per feature map intermedie

    int max_channels = model->input_dim;
    if (model->hidden_dim > max_channels) max_channels = model->hidden_dim;
    if (model->output_dim > max_channels) max_channels = model->output_dim;

    // Alloca entrambi i buffer con max_channels
    float *buffer_in = (float *)malloc(sizeof(float) * batch * max_channels * time_length);
    float *buffer_out = (float *)malloc(sizeof(float) * batch * max_channels * time_length);

    // Copia i dati di input nella parte iniziale di buffer_in.
    // Se input_dim < max_channels, gli elementi in più non vengono usati, ma non causano problemi.
    memcpy(buffer_in, input, sizeof(float)*batch*model->input_dim*time_length);

    // Passo attraverso i layer
    for (int i = 0; i < model->num_layers; i++) {
        if (INFO_MODE) printf("Layer %d: ", i);
        int in_ch = model->layers[i].in_channels;
        int out_ch = model->layers[i].out_channels;
        causal_conv1d_forward(&model->layers[i], buffer_in, buffer_out, batch, in_ch, time_length, INFO_MODE);

        // Applico ReLU tranne che per l'ultimo layer
        if (i < model->num_layers - 1) {
            relu_inplace(buffer_out, batch*out_ch*time_length);
        }

        // Scambio buffer
        if (i < model->num_layers - 1) {
            // Copio buffer_out in buffer_in per il prossimo layer
            float *tmp = buffer_in;
            buffer_in = buffer_out;
            buffer_out = tmp;
        } else {
            // Ultimo layer è direttamente l'output
        }
    }

    // Ora buffer_out contiene l'output dell'ultimo layer
    // Output finale: [batch, output_dim], ultimo timestep
    for (int b = 0; b < batch; b++) {
        for (int od = 0; od < model->output_dim; od++) {
            output[b*model->output_dim + od] = buffer_out[b*model->output_dim*time_length + od*time_length + (time_length-1)];
        }
    }

    free(buffer_in);
    free(buffer_out);
}

