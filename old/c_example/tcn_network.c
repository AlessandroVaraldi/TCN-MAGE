// File: tcn_network.c

#include "tcn_network.h"
#include <stdio.h>
#include <stdlib.h>

void initialize_tcn_network(TCNNetwork* network, int num_layers, int input_channels, int output_channels, int sequence_length) {
    network->num_layers = num_layers;
    network->input_channels = input_channels;
    network->output_channels = output_channels;
    network->sequence_length = sequence_length;
    network->layers = (TCNLayer*)malloc(num_layers * sizeof(TCNLayer));

    if (network->layers == NULL) {
        fprintf(stderr, "Failed to allocate memory for TCN layers\n");
        exit(EXIT_FAILURE);
    }

    int channels = input_channels;
    for (int i = 0; i < num_layers; i++) {
        printf("Initializing layer %d\n", i);
        printf("Input channels: %d\n", channels);
        printf("Output channels: %d\n", (i == num_layers - 1) ? output_channels : channels * 2);
        printf("Kernel size: 3\n");
        printf("Dilation: %d\n", 1 << i);
        printf("\n");
        int out_channels = (i == num_layers - 1) ? output_channels : channels * 2;
        initialize_tcn_layer(&network->layers[i], channels, out_channels, 3, 1 << i);
        channels = out_channels;
    }
}

void apply_tcn_network(TCNNetwork* network, double* input_sequence, double* output_sequence) {
    double* current_input = input_sequence;
    double* current_output = (double*)malloc(network->sequence_length * network->layers[0].c_out * sizeof(double));
    
    if (current_output == NULL) {
        fprintf(stderr, "Failed to allocate memory for initial TCN output\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < network->num_layers; i++) {
        apply_tcn_layer(&network->layers[i], current_input, current_output, network->sequence_length);

        if (i < network->num_layers - 1) {
            double* next_output = (double*)malloc(network->sequence_length * network->layers[i + 1].c_out * sizeof(double));
            if (next_output == NULL) {
                fprintf(stderr, "Failed to allocate memory for intermediate TCN outputs\n");
                free(current_output);
                exit(EXIT_FAILURE);
            }
            if (current_input != input_sequence) {
                free(current_input); // Free the previous output buffer
            }
            current_input = current_output;
            current_output = next_output;
        } else {
            if (current_input != input_sequence) {
                free(current_input);
            }
            memcpy(output_sequence, current_output, network->sequence_length * network->layers[i].c_out * sizeof(double));
            free(current_output);
        }
    }
}

void free_tcn_network(TCNNetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        free(network->layers[i].weights);
        free(network->layers[i].biases);
    }
    free(network->layers);
}


