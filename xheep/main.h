/**
 * @file main.h
 * @brief Header file for example data processing from flash application.
 * 
 * This file contains the necessary includes, definitions, and function 
 * declarations for the example data processing from flash application.
 * 
 * @author Alessandro Varaldi
 * @date 21/01/2025
 * 
 */

#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "x-heep.h"
#include "w25q128jw.h"
#include "core_v_mini_mcu.h"

// === Network Parameters ===
#include "float32/input_output_float32.h"
#include "tcn_network_params.h"
#define BATCH_SIZE  1

// === Precision Types ===
#define PRECISION_FLOAT32 1
#define PRECISION_INT8    2
#define PRECISION_INT16   3
#define PRECISION_INT32   4

#define PRECISION PRECISION_INT8

// === Macros ===
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if PRECISION == PRECISION_FLOAT32
    #include "float32/weights_bias_float32.h"
    #define SCALE 1
    typedef float dtype;
#elif PRECISION == PRECISION_INT8
    #include "int8/weights_bias_int8.h"
    #define SCALE (int)pow(2, FIXED_POINT >> 2)
    typedef int8_t dtype;
#elif PRECISION == PRECISION_INT16
    #include "int16/weights_bias_int16.h"
    #define SCALE (int)pow(2, FIXED_POINT >> 1)
    typedef int16_t dtype;
#elif PRECISION == PRECISION_INT32
    #include "int32/weights_bias_int32.h"
    #define SCALE (int)pow(2, FIXED_POINT)
    typedef int32_t dtype;
#else
    #error "Tipo di precisione non supportato!"
#endif

// === Global Variables ===
static dtype buffer_weights[MAX_HIDDEN_DIM * MAX_HIDDEN_DIM * 3] = {0};
static dtype buffer_bias[MAX_HIDDEN_DIM] = {0};
static dtype intermediate_a[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH] = {0};
static dtype intermediate_b[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH] = {0};

// === Function Prototypes ===
int initialize_input(float (*input)[INPUT_DIM][TIME_LENGTH]);
int quantize_input(float (*input)[INPUT_DIM][TIME_LENGTH], dtype (*q_input)[INPUT_DIM][TIME_LENGTH]);
int inference(float (*q_input)[INPUT_DIM][TIME_LENGTH], float (*q_output)[NUM_CLASSES]);
int dequantize_output(dtype (*q_output)[NUM_CLASSES], float (*output)[NUM_CLASSES]);

static int dequantize_intermediate_buffer(
    dtype (*quant_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    float (*float_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    int output_dim,
    int current_time_length
);

static int requantize_intermediate_buffer(
    float (*float_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    dtype (*quant_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH],
    int output_dim,
    int current_time_length
);

static void printFloat(float number, int decimalPlaces);
static void printIntegerPart(float x);

w25q_error_codes_t fill_buffer(dtype *source, dtype *buffer, int len);
uint32_t heep_get_flash_address_offset(uint32_t* data_address_lma);

// === Function Implementations ===

int check_overflow(int32_t a, int32_t b) {
    int64_t result = (int64_t)a * (int64_t)b;
    return (result < INT32_MIN || result > INT32_MAX);
}

int inference(float (*input)[INPUT_DIM][TIME_LENGTH], float (*output)[NUM_CLASSES])
{
    /**
     * @brief Perform inference on the input data using a Temporal Convolutional Network (TCN).
     *
     * This function processes the input data through multiple layers of a TCN, applying
     * convolution, activation functions, pooling, and other operations as specified by
     * the layer configurations. The final output is stored in the provided output buffer.
     *
     * @param input Pointer to the input data buffer. The input data is expected to be a
     *              3D array with dimensions [BATCH_SIZE][INPUT_DIM][TIME_LENGTH].
     * @param output Pointer to the output data buffer. The output data will be stored as a
     *               2D array with dimensions [BATCH_SIZE][NUM_CLASSES].
     * @return int Returns 0 on success, or EXIT_FAILURE if an error occurs during processing.
     *
     * The function performs the following steps for each layer:
     * 1. Load the weights and biases for the current layer.
     * 2. Perform convolution operations on the input data.
     * 3. Apply bias and store the result in the next buffer.
     * 4. Apply ReLU activation if specified.
     * 5. Apply max pooling if specified.
     * 6. Apply global average pooling if specified.
     * 7. Apply softmax activation if specified.
     * 8. Swap the current and next buffers for the next layer.
     *
     * After processing all layers, the final output is copied to the output buffer.
     */

    int time_length  = TIME_LENGTH; 

    dtype (*current_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_a;
    dtype (*next_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH]    = intermediate_b;
    float (*dq_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH];

    dtype result, input_value, weight;

    float max_val, sum;

    int weight_offset = 0;
    int bias_offset   = 0;
    int layer, b, t, out, i, k, index;
    int input_dim, output_dim, kernel_size, dilation;

    for (layer = 0; layer < NUM_LAYERS; ++layer) {
        printf("Layer %d...\n", layer);
        input_dim   = (layer == 0) ? INPUT_DIM : HIDDEN_DIMS[layer - 1];
        output_dim  = HIDDEN_DIMS[layer];
        kernel_size = KERNEL_SIZES[layer];
        dilation    = DILATIONS[layer];

        if(fill_buffer(&WEIGHTS[weight_offset], buffer_weights, input_dim * output_dim * kernel_size)!=FLASH_OK){
            return EXIT_FAILURE;
        }

        if(fill_buffer(&BIASES[bias_offset], buffer_bias, output_dim)!=FLASH_OK){
            return EXIT_FAILURE;
        }

        // Esegui la convoluzione
        for (b = 0; b < BATCH_SIZE; ++b) {
            for (t = 0; t < time_length; ++t) {
                for (out = 0; out < output_dim; ++out) {
                    result = 0;
                    i = 0;
                    for (i = 0; i < input_dim; ++i) {
                        for (k = 0; k < kernel_size; ++k) {
                            index = t - (kernel_size - 1 - k) * dilation;
                            if (index >= 0 && index < time_length) {
                                input_value = (layer == 0) ? (dtype)(round(input[b][i][index] * SCALE)) : current_buffer[b][i][index];
                                weight = buffer_weights[out * input_dim * kernel_size + i * kernel_size + k];
                                if (check_overflow(input_value, weight)) {
                                    printf("Overflow detected in layer %d\n", layer);
                                    return EXIT_FAILURE;
                                }
                                result += (input_value * weight) / SCALE;
                            }
                        }
                    }
                    result += buffer_bias[out];
                    next_buffer[b][out][t] = result;
                }
            }
        }

        printf("Convolution done\n");

        weight_offset += input_dim * output_dim * kernel_size;
        bias_offset   += output_dim;

        dequantize_intermediate_buffer(next_buffer, dq_buffer, output_dim, time_length);

        printf("Dequantization done\n");

        // Applichiamo la ReLU in virgola mobile
        if (RELU_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        dq_buffer[b][out][t] = (dq_buffer[b][out][t] > 0) ? dq_buffer[b][out][t] : 0.0f;
                    }
                }
            }
            printf("ReLU done\n");
        }


        // Applica Max Pooling se richiesto
        if (MAXPOOL_FLAGS[layer]) {
            time_length = time_length / 2;
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        max_val = dq_buffer[b][out][2 * t];
                        if (2 * t + 1 < TIME_LENGTH) {
                            max_val = MAX(max_val, dq_buffer[b][out][2 * t + 1]);
                        }
                        dq_buffer[b][out][t] = max_val;
                    }
                }
            }
            printf("Max Pooling done\n");
        }

        // Applica Global Average Pooling se richiesto
        if (GAP_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    sum = 0;
                    for (t = 0; t < time_length; ++t) {
                        sum += dq_buffer[b][out][t];
                    }
                    dq_buffer[b][out][0] = sum / time_length;
                }
            }
            // Dopo GAP, di solito riduci anche time_length a 1
            time_length = 1;
            printf("Global Average Pooling done\n");
        }

        // Applica Softmax se richiesto
        if (SOFTMAX_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (t = 0; t < time_length; ++t) {
                    sum = 0;
                    // Prima converti in float, calcoli exp, poi ricasti a dtype se serve
                    // Oppure fai direttamente in float:
                    float float_sum = 0.0f;
                    float float_vals[MAX_HIDDEN_DIM];

                    for (out = 0; out < output_dim; ++out) {
                        float_vals[out] = exp((float)dq_buffer[b][out][t]);
                        float_sum += float_vals[out];
                    }
                    for (out = 0; out < output_dim; ++out) {
                        float_vals[out] /= float_sum;
                        // Se vuoi tenerlo in quantizzato, rimoltiplica e ricasta
                        // altrimenti lascia in float
                        dq_buffer[b][out][t] = float_vals[out];
                    }
                }
            }
            printf("Softmax done\n");
        }

        requantize_intermediate_buffer(dq_buffer, current_buffer, output_dim, time_length);

        printf("Requantization done\n");

        // Clear weights and bias buffers
        for (i = 0; i < input_dim * output_dim * kernel_size; ++i) {
            buffer_weights[i] = 0;
        }
        printf("Cleared weights\n");
        for (i = 0; i < output_dim; ++i) {
            buffer_bias[i] = 0;
        }
        printf("Cleared bias\n");
    }

    // Copia l’uscita finale in "output" (in float)
    for (b = 0; b < BATCH_SIZE; ++b) {
        for (out = 0; out < NUM_CLASSES; ++out) {
            output[b][out] = (float)current_buffer[b][out][0] / SCALE;
        }
    }
    return 0;
}

int initialize_input(float (*input)[INPUT_DIM][TIME_LENGTH]) {
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            for (int k = 0; k < TIME_LENGTH; ++k) {
                input[i][j][k] = REF_INPUT[i * INPUT_DIM * TIME_LENGTH + j * TIME_LENGTH + k];
            }
        }
    }
    return 0;
}


int quantize_input(float (*input)[INPUT_DIM][TIME_LENGTH], dtype (*q_input)[INPUT_DIM][TIME_LENGTH]){
    /**
     * @brief Quantizes the input data by scaling and casting to a specified data type.
     *
     * This function takes a 3D array of floating-point input data and quantizes it by
     * scaling each element by a predefined SCALE factor and casting it to the dtype type.
     * The quantized data is stored in the provided q_input array.
     *
     * @param input A pointer to the 3D array of input data to be quantized. The array
     *              dimensions are [BATCH_SIZE][INPUT_DIM][TIME_LENGTH].
     * @param q_input A pointer to the 3D array where the quantized data will be stored.
     *                The array dimensions are [BATCH_SIZE][INPUT_DIM][TIME_LENGTH].
     * @return Always returns 0.
     */
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            for (int k = 0; k < TIME_LENGTH; ++k) {
                q_input[i][j][k] = (dtype)(input[i][j][k] * SCALE);
            }
        }
    }
    return 0;
}


int dequantize_output(dtype (*q_output)[NUM_CLASSES], float (*output)[NUM_CLASSES]){
    /**
     * @brief Dequantizes the output values.
     *
     * This function takes a quantized output array and dequantizes it by dividing each element
     * by a predefined scale factor. The dequantized values are stored in the provided output array.
     *
     * @param q_output A pointer to the quantized output array with dimensions [BATCH_SIZE][NUM_CLASSES].
     * @param output A pointer to the output array where the dequantized values will be stored, with dimensions [BATCH_SIZE][NUM_CLASSES].
     * @return Always returns 0.
     */
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            output[i][j] = (float)q_output[i][j] / SCALE;
        }
    }
    return 0;
}


static void printIntegerPart(float x) {
    /**
     * @brief Recursively prints the integer part of a floating-point number.
     *
     * This function takes a floating-point number and prints its integer part
     * digit by digit. It handles numbers less than 10 directly and uses recursion
     * for numbers 10 and greater.
     *
     * @param x The floating-point number whose integer part is to be printed.
     */
    if (x < 10.0f) {
        putchar('0' + x);
        return;
    }
    float leading = floorf(x / 10.0f);
    float digit   = fmodf(x, 10.0f);

    printIntegerPart(leading);

    putchar('0' + digit);
}


static void printFloat(float number, int decimalPlaces) {
    /**
     * @brief Prints a floating-point number with a specified number of decimal places.
     *
     * This function takes a floating-point number and prints it to the standard output
     * with the specified number of decimal places. It handles negative numbers by printing
     * a '-' sign and then converting the number to its positive equivalent for further processing.
     *
     * @param number The floating-point number to be printed.
     * @param decimalPlaces The number of decimal places to print.
     */

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
        fracPart = fracPart * 10.0f;
        float digit = floorf(fracPart);
        putchar('0' + digit);
        fracPart = fracPart - digit;
    }
}

// *** MODIFICA INIZIO ***
// Funzione aggiuntiva per dequantizzare un buffer 3D
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

// Funzione aggiuntiva per ri-quantizzare il buffer 3D
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
                // Cast a 'dtype' dopo aver moltiplicato per SCALE
                float val = float_buffer[b][out][t] * SCALE;
                quant_buffer[b][out][t] = (dtype)(val);
            }
        }
    }
    return 0;
}

w25q_error_codes_t fill_buffer(dtype *source, dtype *buffer, int len){
    uint32_t source_flash = heep_get_flash_address_offset((uint32_t*)source);
    w25q_error_codes_t status = w25q128jw_read_standard(source_flash, buffer, (uint32_t) len*4);
    return status;
}

#endif // DATA_H_
