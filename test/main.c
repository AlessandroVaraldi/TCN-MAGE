#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "main.h"

//float input[BATCH_SIZE][INPUT_DIM][TIME_LENGTH] = {0};
float output[BATCH_SIZE][NUM_CLASSES] = {0};

// === Function Prototypes ===
int inference(float (*output)[NUM_CLASSES]);
int compare_output(const float (*output)[NUM_CLASSES]);

int main(int argc, char *argv[]) {

    printf("Esecuzione dell'inferenza...\n");
    if (inference(output) != EXIT_SUCCESS) {
        printf("Errore inferenza\n");
        return EXIT_FAILURE;
    }

    printf("Confronto con output di riferimento...\n");
    compare_output(output);

    return EXIT_SUCCESS;
}

int compare_output(const float (*output)[NUM_CLASSES]) {
    float reference_output[BATCH_SIZE][NUM_CLASSES];
    float current_output;
    float current_reference_output;

    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            reference_output[i][j] = REF_OUTPUT[i * NUM_CLASSES + j];
        }
    }

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < NUM_CLASSES; ++o) {
            current_output = output[b][o];
            current_reference_output = reference_output[b][o];

            printf("Output Python: ");
            printFloat((float)current_reference_output, 6);
            printf(", Output C: ");
            printFloat((float)current_output, 6);
            printf(", ");
            if (fabsf((float)current_output - (float)current_reference_output) > 1e-1) {
                printf("Errore: i valori non corrispondono!\n");
            }
            else {
                printf("Valori corrispondenti.\n");
            }
        }
    }
    return 0;
}
