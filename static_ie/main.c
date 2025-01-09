#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// === Precision Types ===
#define PRECISION_FLOAT32 1
#define PRECISION_INT8    2
#define PRECISION_INT16   3
#define PRECISION_INT32   4

// === Network Parameters ===
#define INPUT_DIM   4
#define OUTPUT_DIM  1
#define HIDDEN_DIM  2
#define NUM_LAYERS  3
#define KERNEL_SIZE 3
#define BATCH_SIZE  1
#define TIME_LENGTH 8
#define PRECISION   PRECISION_INT16

// === Macros ===
#define SCALE ((PRECISION == PRECISION_FLOAT32) ? 1 : \
              (PRECISION == PRECISION_INT8)   ? (1 << 4) : \
              (PRECISION == PRECISION_INT16)  ? (1 << 8) : \
              (PRECISION == PRECISION_INT32)  ? (1 << 16) : 1)

#if PRECISION == PRECISION_FLOAT32
    #include "tcn_weights_float32.h"
    #include "tcn_input_output_float32.h"
    typedef float dtype;
#elif PRECISION == PRECISION_INT8
    #include "tcn_weights_int8.h"
    #include "tcn_input_output_int8.h"
    typedef int8_t dtype;
#elif PRECISION == PRECISION_INT16
    #include "tcn_weights_int16.h"
    #include "tcn_input_output_int16.h"
    typedef int16_t dtype;
#elif PRECISION == PRECISION_INT32
    #include "tcn_weights_int32.h"
    #include "tcn_input_output_int32.h"
    typedef int32_t dtype;
#else
    #error "Tipo di precisione non supportato!"
#endif

// === Static Buffers ===
static dtype input[BATCH_SIZE][INPUT_DIM][TIME_LENGTH];
static dtype output[BATCH_SIZE][OUTPUT_DIM];
static dtype intermediate_a[BATCH_SIZE][HIDDEN_DIM][TIME_LENGTH];
static dtype intermediate_b[BATCH_SIZE][HIDDEN_DIM][TIME_LENGTH];

// === Function Prototypes ===
void initialize_input();
void inference();
void compare_output();

// === Function Implementations ===
void initialize_input() {
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            for (int k = 0; k < TIME_LENGTH; ++k) {
                input[i][j][k] = INPUT_REF[i * INPUT_DIM * TIME_LENGTH + j * TIME_LENGTH + k];
            }
        }
    }
}

void inference() {
    dtype (*current_buffer)[HIDDEN_DIM][TIME_LENGTH] = intermediate_a;
    dtype (*next_buffer)[HIDDEN_DIM][TIME_LENGTH] = intermediate_b;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        int dilation = (layer == 0) ? 1 : (1 << layer);

        const dtype *current_weights;
        const dtype *current_bias;

        int input_dim = (layer == 0) ? INPUT_DIM : HIDDEN_DIM;
        int output_dim = (layer == NUM_LAYERS - 1) ? OUTPUT_DIM : HIDDEN_DIM;

        if (layer == 0) {
            current_weights = (const dtype *)layer0_weights;
            current_bias = layer0_biases;
        } else if (layer == NUM_LAYERS - 1) {
            current_weights = (const dtype *)layer2_weights;
            current_bias = layer2_biases;
        } else {
            current_weights = (const dtype *)layer1_weights;
            current_bias = layer1_biases;
        }

        for (int batch = 0; batch < BATCH_SIZE; ++batch) {
            for (int time = 0; time < TIME_LENGTH; ++time) {
                for (int o = 0; o < output_dim; ++o) {
                    dtype result = 0;

                    for (int i = 0; i < input_dim; ++i) {
                        for (int k = 0; k < KERNEL_SIZE; ++k) {
                            int weight_index = o * input_dim * KERNEL_SIZE + i * KERNEL_SIZE + k;
                            int input_time = time - (KERNEL_SIZE - 1 - k) * dilation;

                            if (input_time >= 0 && input_time < TIME_LENGTH) {
                                dtype input_value = (layer == 0) ? input[batch][i][input_time] : current_buffer[batch][i][input_time];
                                result += current_weights[weight_index] * input_value / SCALE;
                            }
                        }
                    }

                    result += current_bias[o];

                    if (layer != NUM_LAYERS - 1 && result < 0) {
                        result = (result > 0) ? result : 0; // Calcolo diretto
                    }

                    if (layer == NUM_LAYERS - 1) {
                        output[batch][o] = result;
                    } else {
                        next_buffer[batch][o][time] = result;
                    }
                }
            }
        }

        dtype (*temp)[HIDDEN_DIM][TIME_LENGTH] = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
    }
}

void compare_output() {
    dtype reference_output[BATCH_SIZE][OUTPUT_DIM];

    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_DIM; ++j) {
            reference_output[i][j] = OUTPUT_REF[i * OUTPUT_DIM + j];
        }
    }

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < OUTPUT_DIM; ++o) {
            printf("Output C: %d, Output Python: %d\n", output[b][o] / SCALE, reference_output[b][o] / SCALE);

            if (abs(output[b][o] - reference_output[b][o]) > 1) {
                printf("Discrepanza trovata!\n");
                return;
            }
        }
    }

    printf("I risultati coincidono!\n");
}

int main() {
    printf("Inizializzazione dell'input...\n");
    initialize_input();

    printf("Esecuzione dell'inferenza...\n");
    inference();

    printf("Confronto con output di riferimento...\n");
    compare_output();

    return 0;
}
