#include "tcn_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

// Funzione per generare una sequenza deterministica, ad esempio una funzione sinusoidale
void generate_deterministic_input(double* input_sequence, int sequence_length, int input_channels) {
    for (int i = 0; i < sequence_length; i++) {
        // Esempio di funzione sinusoidale come input deterministico
        input_sequence[i] = sin(2 * PI * i / sequence_length);
    }
}

int main() {
    int num_layers = 8;        // Numero di layer nella rete
    int input_channels = 1;    // Canali di ingresso (ad es. unidimensionali per dati di vibrazione)
    int output_channels = 1;   // Canali di uscita (ad es. predizione della prossima vibrazione)
    int sequence_length = 16;  // Lunghezza della sequenza di input/output

    // Crea la sequenza di input e output (inizializzati a zero)
    double* input_sequence = (double*)calloc(sequence_length * input_channels, sizeof(double));
    double* output_sequence = (double*)calloc(sequence_length * output_channels, sizeof(double));

    // Verifica l'allocazione della memoria
    if (input_sequence == NULL || output_sequence == NULL) {
        fprintf(stderr, "Failed to allocate memory for input or output sequences\n");
        return EXIT_FAILURE;
    }

    // Genera un input deterministico usando una funzione sinusoidale
    generate_deterministic_input(input_sequence, sequence_length, input_channels);

    // Inizializzazione della rete TCN
    TCNNetwork network;
    initialize_tcn_network(&network, num_layers, input_channels, output_channels, sequence_length);

    // Applicazione della rete TCN sull'input
    apply_tcn_network(&network, input_sequence, output_sequence);

    // Stampa dell'output in output.txt
    FILE* output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Failed to open output file\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < sequence_length; i++) {
        fprintf(output_file, "%f\n", output_sequence[i]);
    }

    // Liberazione della memoria
    free_tcn_network(&network);
    free(input_sequence);
    free(output_sequence);

    return EXIT_SUCCESS;
}
