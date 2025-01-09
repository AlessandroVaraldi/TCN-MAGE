#include <stdio.h>
#include <stdlib.h>
#include "tcn_weights_float32.h"  // Sostituisci con la precisione desiderata
#include "tcn_input_output_ref.h"  // File aggiunto per input/output di riferimento

#define SEQUENCE_LENGTH 8
#define INPUT_DIM 4
#define OUTPUT_DIM 1

void tcn_inference(const float input[SEQUENCE_LENGTH][INPUT_DIM], float output[OUTPUT_DIM]) {
    // Esegui inferenza TCN usando i pesi e bias inclusi

    // Primo layer
    static float layer_output[SEQUENCE_LENGTH][32]; // 32 è un esempio di hidden_dim
    for (int i = 0; i < 32; i++) {
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            float sum = layer0_biases[i];
            for (int k = 0; k < INPUT_DIM; k++) {
                sum += input[t][k] * layer0_weights[i * INPUT_DIM + k];
            }
            layer_output[t][i] = sum > 0 ? sum : 0; // ReLU attivazione
        }
    }

    // Secondo layer
    static float final_output[SEQUENCE_LENGTH][OUTPUT_DIM];
    for (int i = 0; i < OUTPUT_DIM; i++) {
        for (int t = 0; t < SEQUENCE_LENGTH; t++) {
            float sum = layer1_biases[i];
            for (int k = 0; k < 32; k++) {
                sum += layer_output[t][k] * layer1_weights[i * 32 + k];
            }
            final_output[t][i] = sum;
        }
    }

    // Estrai l'output finale (ultimo timestamp)
    for (int i = 0; i < OUTPUT_DIM; i++) {
        output[i] = final_output[SEQUENCE_LENGTH - 1][i];
    }
}

int main() {
    // Input di test preso dal file di riferimento
    float input[SEQUENCE_LENGTH][INPUT_DIM] = INPUT_REF;

    float output[OUTPUT_DIM];
    tcn_inference(input, output);

    // Stampa l'output
    for (int i = 0; i < OUTPUT_DIM; i++) {
        printf("Output[%d]: %f\n", i, output[i]);
    }

    // Verifica con output di riferimento
    int correct = 1;
    for (int i = 0; i < OUTPUT_DIM; i++) {
        if (fabs(output[i] - OUTPUT_REF[i]) > 1e-5) {
            correct = 0;
            printf("Errore: Output[%d] = %f, Atteso = %f\n", i, output[i], OUTPUT_REF[i]);
        }
    }

    if (correct) {
        printf("Inferenza corretta!\n");
    } else {
        printf("Inferenza errata!\n");
    }

    return 0;
}
