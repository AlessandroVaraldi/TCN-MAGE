/**
 * @file main.h
 * @brief Header file for example data processing from flash application.
 *
 * Questa versione unifica i vari buffer globali (pesi, bias, intermediate_a/b)
 * in un unico array globale "global_buffer", suddiviso a offset.
 */

#ifndef MAIN_H_
#define MAIN_H_

#define XHEEP 0

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#if XHEEP
#include "x-heep.h"
#include "w25q128jw.h"
#include "dma_sdk.h"
#endif

// === Network Parameters ===
#include "input_output.h"
#include "tcn_network_params.h"

// Per semplicità, rimane BATCH_SIZE=1
#define BATCH_SIZE  1

// === Precision Types ===
#define PRECISION_FLOAT32 1
#define PRECISION_INT8    2
#define PRECISION_INT16   3
#define PRECISION_INT32   4

#define PRECISION PRECISION_FLOAT32

// === Macros ===
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if PRECISION == PRECISION_FLOAT32
    #include "weights_bias_float32.h"
    #define SCALE 1
    typedef float dtype;
    typedef float double_dtype;
#elif PRECISION == PRECISION_INT8
    #include "weights_bias_int8.h"
    #define SCALE pow(2, FIXED_POINT >> 2)
    typedef int8_t dtype;
    typedef int16_t double_dtype;
#elif PRECISION == PRECISION_INT16
    #include "weights_bias_int16.h"
    #define SCALE pow(2, FIXED_POINT >> 1)
    typedef int16_t dtype;
    typedef int32_t double_dtype;
#elif PRECISION == PRECISION_INT32
    #include "weights_bias_int32.h"
    #define SCALE pow(2, FIXED_POINT)
    typedef int32_t dtype;
    typedef int64_t double_dtype;
#else
    #error "Tipo di precisione non supportato!"
#endif

#define SIZE_WEIGHTS      (MAX_HIDDEN_DIM * 3)
#define SIZE_BIAS         (1)
#define SIZE_INTERMEDIATE (BATCH_SIZE * MAX_HIDDEN_DIM * TIME_LENGTH)

#define GLOBAL_BUFFER_SIZE ( SIZE_WEIGHTS + SIZE_BIAS + 2 * SIZE_INTERMEDIATE )

static dtype global_buffer[GLOBAL_BUFFER_SIZE] = {0};

float dequantized_buffer[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH];

#define OFFSET_WEIGHTS        0
#define OFFSET_BIAS           (OFFSET_WEIGHTS + SIZE_WEIGHTS)
#define OFFSET_INTERMEDIATE_A (OFFSET_BIAS + SIZE_BIAS)
#define OFFSET_INTERMEDIATE_B (OFFSET_INTERMEDIATE_A + SIZE_INTERMEDIATE)

static dtype *buffer_weights = &global_buffer[OFFSET_WEIGHTS];
static dtype *buffer_bias    = &global_buffer[OFFSET_BIAS];

static dtype (*intermediate_a)[MAX_HIDDEN_DIM][TIME_LENGTH] =
    (dtype (*)[MAX_HIDDEN_DIM][TIME_LENGTH]) &global_buffer[OFFSET_INTERMEDIATE_A];

static dtype (*intermediate_b)[MAX_HIDDEN_DIM][TIME_LENGTH] =
    (dtype (*)[MAX_HIDDEN_DIM][TIME_LENGTH]) &global_buffer[OFFSET_INTERMEDIATE_B];


// ========================================================================
// Function Prototypes
// ========================================================================

/**
 * @brief Perform the entire inference process and store the output in float form.
 *        The final output dimension is [BATCH_SIZE][NUM_CLASSES].
 */
int inference(float (*output)[NUM_CLASSES]);

/**
 * @brief Dequantize a 3D buffer from quant_buffer -> float_buffer
 */
static int dequantize_intermediate_buffer(
    dtype (*quant_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    float (*float_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    int output_dim,
    int current_time_length
);

/**
 * @brief Re-quantize a 3D buffer from float_buffer -> quant_buffer
 */
static int requantize_intermediate_buffer(
    float (*float_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    dtype (*quant_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    int output_dim,
    int current_time_length
);

/**
 * @brief Print a float number with specified decimal places.
 */
static void printFloat(float number, int decimalPlaces);

/**
 * @brief Print the integer part of a float (recursive).
 */
static void printIntegerPart(float x);

#if XHEEP
/**
 * @brief Fill a buffer in SRAM by reading from SPI flash (standard read).
 */
w25q_error_codes_t fill_buffer(dtype *source, dtype *buffer, int len);

/**
 * @brief Get the flash address offset from LMA pointer.
 */
uint32_t heep_get_flash_address_offset(uint32_t* data_address_lma);
#endif

// ========================================================================
// Function Implementations
// ========================================================================

/**
 * @brief Dequantize a 3D buffer from quant_buffer -> float_buffer
 */
static int dequantize_intermediate_buffer(
    dtype (*quant_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    float (*float_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    int output_dim,
    int current_time_length
)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int out = 0; out < output_dim; ++out) {
            for (int t = 0; t < current_time_length; ++t) {
                float_buffer[b][out][t] = (float)quant_buffer[b][out][t] / SCALE;
            }
        }
    }
    return 0;
}

/**
 * @brief Re-quantize a 3D buffer from float_buffer -> quant_buffer
 */
static int requantize_intermediate_buffer(
    float (*float_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    dtype (*quant_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    int output_dim,
    int current_time_length
)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int out = 0; out < output_dim; ++out) {
            for (int t = 0; t < current_time_length; ++t) {
                float val = float_buffer[b][out][t] * SCALE;
                quant_buffer[b][out][t] = (dtype)val;
            }
        }
    }
    return 0;
}

/**
 * @brief Print the integer part of a float (recursively).
 */
static void printIntegerPart(float x) {
    if (x < 10.0f) {
        putchar('0' + (int)x);
        return;
    }
    float leading = floorf(x / 10.0f);
    float digit   = fmodf(x, 10.0f);

    printIntegerPart(leading);
    putchar('0' + (int)digit);
}

/**
 * @brief Print a float number with the specified number of decimal places.
 */
static void printFloat(float number, int decimalPlaces) {
    if (number < 0.0f) {
        putchar('-');
        number = -number;
    }

    float intPart;
    float fracPart = modff(number, &intPart);

    if (intPart == 0.0f) {
        putchar('0');
    } else {
        printIntegerPart(intPart);
    }

    putchar('.');

    for (int i = 0; i < decimalPlaces; i++) {
        fracPart *= 10.0f;
        float digit = floorf(fracPart);
        putchar('0' + (int)digit);
        fracPart -= digit;
    }
}

#if XHEEP
/**
 * @brief Fill a buffer in SRAM by reading from SPI flash.
 */
w25q_error_codes_t fill_buffer(dtype *source, dtype *buffer, int len){
    uint32_t source_flash = heep_get_flash_address_offset((uint32_t*)source);
    w25q_error_codes_t status = w25q128jw_read_standard(
        source_flash,
        buffer,
        (uint32_t) len * sizeof(dtype)
    );
    return status;
}
#endif

/**
 * @brief Perform inference on the input data and produce output probabilities.
 *
 * This function performs a series of convolutional, activation, pooling, and normalization
 * operations on the input data to produce output probabilities for each class.
 *
 * @param output Pointer to the output array where the final probabilities will be stored.
 *               The array should have dimensions [BATCH_SIZE][NUM_CLASSES].
 * @return int Returns 0 on success, or EXIT_FAILURE if an error occurs.
 *
 * The function follows these steps:
 * 1. Precompute SCALEd REF_INPUT values.
 * 2. Initialize the current buffer with SCALEd input values.
 * 3. For each layer in the network:
 *    - Perform convolution with the appropriate weights and biases.
 *    - Dequantize the intermediate buffer.
 *    - Apply ReLU activation if specified.
 *    - Apply MaxPooling if specified.
 *    - Apply Global Average Pooling (GAP) if specified.
 *    - Apply Softmax normalization if specified.
 *    - Requantize the intermediate buffer.
 * 4. Store the final output probabilities in the output array.
 *
 * The function uses several macros and constants:
 * - BATCH_SIZE: The number of input samples processed in parallel.
 * - INPUT_DIM: The dimensionality of the input data.
 * - TIME_LENGTH: The length of the time dimension in the input data.
 * - NUM_CLASSES: The number of output classes.
 * - NUM_LAYERS: The number of layers in the network.
 * - MAX_HIDDEN_DIM: The maximum dimensionality of the hidden layers.
 * - SCALE: The scaling factor used for quantization.
 * - REF_INPUT: The reference input data.
 * - HIDDEN_DIMS: Array specifying the dimensionality of each hidden layer.
 * - KERNEL_SIZES: Array specifying the kernel size for each layer.
 * - DILATIONS: Array specifying the dilation factor for each layer.
 * - RELU_FLAGS: Array specifying whether to apply ReLU activation for each layer.
 * - MAXPOOL_FLAGS: Array specifying whether to apply MaxPooling for each layer.
 * - GAP_FLAGS: Array specifying whether to apply Global Average Pooling for each layer.
 * - SOFTMAX_FLAGS: Array specifying whether to apply Softmax normalization for each layer.
 * - WEIGHTS: Array containing the weights for each layer.
 * - BIASES: Array containing the biases for each layer.
 * - FLASH_OK: Constant indicating successful flash memory operation.
 * - EXIT_FAILURE: Constant indicating failure.
 *
 * The function also uses several helper functions:
 * - fill_buffer: Fills a buffer with data from flash memory.
 * - dequantize_intermediate_buffer: Dequantizes the intermediate buffer.
 * - requantize_intermediate_buffer: Requantizes the intermediate buffer.
 */
int inference(float (*output)[NUM_CLASSES])
{
    dtype (*current_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_a;
    dtype (*next_buffer)  [MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_b;

    int time_length  = TIME_LENGTH;
    int layer, out, b, t, i, k;
    int input_dim, output_dim, kernel_size, dilation;

    int weight_offset = 0;
    int bias_offset   = 0;

    // Precompute SCALEd REF_INPUT values
    dtype scaled_ref_input[BATCH_SIZE * INPUT_DIM * TIME_LENGTH];
    for (int i = 0; i < BATCH_SIZE * INPUT_DIM * TIME_LENGTH; ++i) {
        scaled_ref_input[i] = (dtype) REF_INPUT[i] * SCALE;
    }

    // Current buffer initialization
    for (b = 0; b < BATCH_SIZE; ++b) {
        for (out = 0; out < INPUT_DIM; ++out) {
            for (t = 0; t < TIME_LENGTH; ++t) {
                current_buffer[b][out][t] = scaled_ref_input[b * INPUT_DIM * TIME_LENGTH + out * TIME_LENGTH + t];
            }
        }
    }

    for (layer = 0; layer < NUM_LAYERS; ++layer) {
        printf("Layer %d...\n", layer);

        input_dim   = (layer == 0) ? INPUT_DIM : HIDDEN_DIMS[layer - 1];
        output_dim  = HIDDEN_DIMS[layer];
        kernel_size = KERNEL_SIZES[layer];
        dilation    = DILATIONS[layer];

        for (out = 0; out < output_dim; ++out) {
            #if XHEEP
            if (fill_buffer(&WEIGHTS[weight_offset], buffer_weights, input_dim * kernel_size) != FLASH_OK)
            {
                return EXIT_FAILURE;
            }
            #else
            for (i = 0; i < input_dim; ++i) {
                for (k = 0; k < kernel_size; ++k) {
                    buffer_weights[i*kernel_size + k] = WEIGHTS[weight_offset + i*kernel_size + k];
                }
            }
            #endif
            weight_offset += input_dim * kernel_size;

            #if XHEEP
            if (fill_buffer(&BIASES[bias_offset], buffer_bias, 1) != FLASH_OK)
            {
                return EXIT_FAILURE;
            }
            #else
            for (i = 0; i < SIZE_BIAS; ++i) {
                buffer_bias[i] = BIASES[bias_offset + i];
            }
            #endif
            bias_offset += 1;

            for (b = 0; b < BATCH_SIZE; ++b) {
                for (t = 0; t < time_length; ++t) {
                    double_dtype result = 0;
                    dtype scaled_result = 0;
                    for (i = 0; i < input_dim; ++i) {
                        for (k = 0; k < kernel_size; ++k) {
                            int index = t - (kernel_size - 1 - k) * dilation;
                            if (index >= 0 && index < time_length) {
                                dtype input_value = current_buffer[b][i][index];
                                dtype weight = buffer_weights[i*kernel_size + k];

                                result = (double_dtype)input_value * (double_dtype)weight;
                                scaled_result += (dtype)(result / SCALE);
                            }
                        }
                    }
                    scaled_result += buffer_bias[0];
                    next_buffer[b][out][t] = scaled_result;
                }
            }
        }
        printf("    Convolution done\n");

        dequantize_intermediate_buffer(next_buffer, dequantized_buffer, output_dim, time_length);
        printf("    Dequantized\n");

        if (RELU_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        if (dequantized_buffer[b][out][t] < 0) {
                            dequantized_buffer[b][out][t] = 0.0f;
                        }
                    }
                }
            }
            printf("    ReLU applied\n");
        }

        if (MAXPOOL_FLAGS[layer]) {
            time_length = time_length / 2;
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        float a_ = dequantized_buffer[b][out][2*t];
                        float b_ = dequantized_buffer[b][out][2*t+1];
                        dequantized_buffer[b][out][t] = (a_ > b_) ? a_ : b_;
                    }
                }
            }
            printf("    MaxPool applied\n");
        }

        if (AVGPOOL_FLAGS[layer]) {
            time_length = time_length / 2;
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        float a_ = dequantized_buffer[b][out][2*t];
                        float b_ = dequantized_buffer[b][out][2*t+1];
                        dequantized_buffer[b][out][t] = (a_ + b_) / 2;
                    }
                }
            }
            printf("    AvgPool applied\n");
        }

        if (GAP_FLAGS[layer]) {
            float sum;
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    sum = 0.0f;
                    for (t = 0; t < time_length; ++t) {
                        sum += dequantized_buffer[b][out][t];
                    }
                    dequantized_buffer[b][out][0] = sum / time_length;
                }
            }
            time_length = 1;
            printf("    GAP applied\n");
        }

        if (SOFTMAX_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (t = 0; t < time_length; ++t) {
                    float sum_ = 0.0f;
                    float tmp[MAX_HIDDEN_DIM];
                    for (out = 0; out < output_dim; ++out) {
                        tmp[out] = expf(dequantized_buffer[b][out][t]);
                        sum_ += tmp[out];
                    }
                    for (out = 0; out < output_dim; ++out) {
                        dequantized_buffer[b][out][t] = tmp[out] / sum_;
                    }
                }
            }
            printf("    Softmax applied\n");
        }

        requantize_intermediate_buffer(dequantized_buffer, current_buffer, output_dim, time_length);
        printf("    Requantized\n");
    }

    for (int b_ = 0; b_ < BATCH_SIZE; ++b_) {
        for (int out_ = 0; out_ < NUM_CLASSES; ++out_) {
            output[b_][out_] = (float)current_buffer[b_][out_][0] / SCALE;
        }
    }
    return 0;
}

#endif // MAIN_H_
