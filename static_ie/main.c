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
#include "tcn_network_params.h"
#define BATCH_SIZE  1

#define PRECISION   PRECISION_FLOAT32

// === Macros ===
#define SCALE ((PRECISION == PRECISION_FLOAT32) ? 1 : \
              (PRECISION == PRECISION_INT8)   ? (1 << FIXED_POINT/4) : \
              (PRECISION == PRECISION_INT16)  ? (1 << FIXED_POINT/2) : \
              (PRECISION == PRECISION_INT32)  ? (1 << FIXED_POINT) : 1)

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if PRECISION == PRECISION_FLOAT32
    #include "float32/weights_bias_float32.h"
    #include "float32/input_output_float32.h"
    typedef float dtype;
#elif PRECISION == PRECISION_INT8
    #include "int8/weights_bias_int8.h"
    #include "int8/input_output_int8.h"
    typedef int8_t dtype;
#elif PRECISION == PRECISION_INT16
    #include "int16/weights_bias_int16.h"
    #include "int16/input_output_int16.h"
    typedef int16_t dtype;
#elif PRECISION == PRECISION_INT32
    #include "int32/weights_bias_int32.h"
    #include "int32/input_output_int32.h"
    typedef int32_t dtype;
#else
    #error "Tipo di precisione non supportato!"
#endif


// === Static Buffers ===
static dtype input[BATCH_SIZE][INPUT_DIM][TIME_LENGTH];
static dtype output[BATCH_SIZE][NUM_CLASSES];
static dtype intermediate_a[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH];
static dtype intermediate_b[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH];

// === Function Prototypes ===
void initialize_input();
void inference();
void compare_output();
void print_float_as_int(float number, int decimal_places);

int main() {
    printf("Inizializzazione dell'input...\n");
    initialize_input();

    printf("Esecuzione dell'inferenza...\n");
    inference();

    printf("Confronto con output di riferimento...\n");
    compare_output();

    return 0;
}

// === Function Implementations ===
void initialize_input() {
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            for (int k = 0; k < TIME_LENGTH; ++k) {
                input[i][j][k] = REF_INPUT[i * INPUT_DIM * TIME_LENGTH + j * TIME_LENGTH + k];
            }
        }
    }
}

void inference() {
    dtype (*current_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_a;
    dtype (*next_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_b;

    int weight_offset = 0;
    int bias_offset = 0;

    // Compute bias offset
    for (int i = 0; i < NUM_LAYERS; ++i) {
        if (i == 0) {
            bias_offset += INPUT_DIM * HIDDEN_DIMS[i] * KERNEL_SIZES[i];
        } else {
            bias_offset += HIDDEN_DIMS[i - 1] * HIDDEN_DIMS[i] * KERNEL_SIZES[i];
        }
    }

    printf("Bias offset: %d\n", bias_offset);

    int num_layers = NUM_LAYERS;

    int time_length = TIME_LENGTH;

    for (int layer = 0; layer < num_layers; ++layer) {
        int input_dim = (layer == 0) ? INPUT_DIM : HIDDEN_DIMS[layer - 1];
        int output_dim = HIDDEN_DIMS[layer];
        int kernel_size = KERNEL_SIZES[layer];
        int dilation = DILATIONS[layer];

        const dtype *current_weights = &WEIGHTS_BIAS[weight_offset];
        const dtype *current_bias = &WEIGHTS_BIAS[bias_offset];

        // Update offsets
        weight_offset += input_dim * output_dim * kernel_size;
        bias_offset += output_dim;

        // Apply causal convolution
        for (int b = 0; b < BATCH_SIZE; ++b) {
            for (int t = 0; t < time_length; ++t) {
                for (int out = 0; out < output_dim; ++out) {
                    dtype result = 0;
                    for (int in = 0; in < input_dim; ++in) {
                        for (int k = 0; k < kernel_size; ++k) {
                            int index = t - (kernel_size - 1 - k)* dilation;
                            if (index >= 0 && index < time_length) {
                                dtype input_value = (layer == 0) ? input[b][in][index] : current_buffer[b][in][index];
                                result += current_weights[out * input_dim * kernel_size + in * kernel_size + k] * input_value / SCALE;
                            }
                        }
                    }
                    result += current_bias[out];
                    next_buffer[b][out][t] = result;
                }
            }
        }

        // Apply ReLU if enabled
        if (RELU_FLAGS[layer]) {
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int out = 0; out < output_dim; ++out) {
                    for (int t = 0; t < time_length; ++t) {
                        next_buffer[b][out][t] = next_buffer[b][out][t] > 0 ? next_buffer[b][out][t] : 0;
                    }
                }
            }
        }

        // Apply MaxPool if enabled
        if (MAXPOOL_FLAGS[layer]) {
            time_length = time_length / 2;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int out = 0; out < output_dim; ++out) {
                    for (int t = 0; t < time_length; ++t) {
                        dtype max_val = next_buffer[b][out][2 * t];
                        if (2 * t + 1 < TIME_LENGTH) {
                            max_val = MAX(max_val, next_buffer[b][out][2 * t + 1]);
                        }
                        next_buffer[b][out][t] = max_val;
                    }
                }
            }
        }

        // Apply GAP if enabled
        if (GAP_FLAGS[layer]) {
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int out = 0; out < output_dim; ++out) {
                    dtype sum = 0;
                    for (int t = 0; t < time_length; ++t) {
                        sum += next_buffer[b][out][t];
                    }
                    next_buffer[b][out][0] = sum / time_length;
                }
            }
        }

        // Apply Softmax if enabled
        if (SOFTMAX_FLAGS[layer]) {
            for (int b = 0; b < BATCH_SIZE; ++b) {
                for (int t = 0; t < time_length; ++t) {
                    dtype sum = 0;
                    for (int out = 0; out < output_dim; ++out) {
                        next_buffer[b][out][t] = exp(next_buffer[b][out][t]);
                        sum += next_buffer[b][out][t];
                    }
                    for (int out = 0; out < output_dim; ++out) {
                        next_buffer[b][out][t] /= sum;
                    }
                }
            }
        }

        // Swap buffers
        if (layer < num_layers - 1) {
            dtype (*temp)[MAX_HIDDEN_DIM][TIME_LENGTH] = current_buffer;
            current_buffer = next_buffer;
            next_buffer = temp;
        }
    }

    // Write final output
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int out = 0; out < NUM_CLASSES; ++out) {
            output[b][out] = next_buffer[b][out][0];
        }
    }
}


void compare_output() {
    dtype reference_output[BATCH_SIZE][NUM_CLASSES];
    dtype current_output;
    dtype current_reference_output;

    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            reference_output[i][j] = REF_OUTPUT[i * NUM_CLASSES + j];
        }
    }

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < NUM_CLASSES; ++o) {
            current_output = output[b][o];
            current_reference_output = reference_output[b][o];

            if (PRECISION == PRECISION_FLOAT32) {
                printf("Output Python: ");
                print_float_as_int((float)current_reference_output, 6);
                printf(", Output C: ");
                print_float_as_int((float)current_output, 6);
                printf(", ");
                if (fabsf((float)current_output - (float)current_reference_output) > 1e-3) {
                    printf("Errore: i valori non corrispondono!\n");
                }
                else {
                    printf("Valori corrispondenti.\n");
                }
            }
            else {
                printf("Output Python: %d, Output C: %d, ", current_reference_output, current_output);
                if (current_output != current_reference_output) {
                    printf("Errore: i valori non corrispondono!\n");
                }
                else {
                    printf("Valori corrispondenti.\n");
                }
            }
        }
    }
}

void print_float_as_int(float number, int decimal_places) {
    if (number < 0) {
        putchar('-');
        number = -number;
    }

    int32_t integer_part = (int32_t)number;
    float decimal_part = number - (float)integer_part;

    printf("%d", integer_part);

    if (decimal_places > 0) {
        putchar('.');
        char buffer[decimal_places + 1];
        snprintf(buffer, sizeof(buffer), "%.*f", decimal_places, decimal_part);
        printf("%s", buffer + 2); // Skip "0."
    }
    if (decimal_places > 0) {
        putchar('.');
        // Adding 0.5f to decimal_part before casting to int32_t for rounding
        int32_t decimal_digits = (int32_t)(decimal_part + 0.5f);

        printf("%0*d", decimal_places, decimal_digits);
    }
}
