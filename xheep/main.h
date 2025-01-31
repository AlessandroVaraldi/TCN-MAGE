/**
 * @file main.h
 * @brief Header file for example data processing from flash application.
 *
 * Questa versione unifica i vari buffer globali (pesi, bias, intermediate_a/b)
 * in un unico array globale "global_buffer", suddiviso a offset.
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
#include "float32/input_output_float32_2.h"
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

#define SIZE_WEIGHTS      (MAX_HIDDEN_DIM * 3)
#define SIZE_BIAS         (1)
#define SIZE_INTERMEDIATE (BATCH_SIZE * MAX_HIDDEN_DIM * TIME_LENGTH)

#define GLOBAL_BUFFER_SIZE ( SIZE_WEIGHTS + SIZE_BIAS + 2 * SIZE_INTERMEDIATE )

static dtype global_buffer[GLOBAL_BUFFER_SIZE] = {0};

float dq_buffer[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH];

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

/**
 * @brief Fill a buffer in SRAM by reading from SPI flash (standard read).
 */
w25q_error_codes_t fill_buffer(dtype *source, dtype *buffer, int len);

/**
 * @brief Get the flash address offset from LMA pointer.
 */
uint32_t heep_get_flash_address_offset(uint32_t* data_address_lma);

/**
 * @brief Check if a * b overflows int32_t range.
 */
int check_overflow(int32_t a, int32_t b);


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

/**
 * @brief Fill a buffer in SRAM by reading from SPI flash.
 */
w25q_error_codes_t fill_buffer(dtype *source, dtype *buffer, int len){
    // Calcola offset in flash
    uint32_t source_flash = heep_get_flash_address_offset((uint32_t*)source);
    // Legge len * sizeof(dtype) byte
    w25q_error_codes_t status = w25q128jw_read_standard(
        source_flash,
        buffer,
        (uint32_t) len * sizeof(dtype)
    );
    return status;
}

/**
 * @brief Esempio di funzione "inference" (estratto semplificato).
 *        Qui elabori i dati layer per layer usando intermediate_a/b,
 *        buffer_weights, buffer_bias, etc.
 */
int inference(float (*output)[NUM_CLASSES])
{
    // Esempio: usiamo i due buffer intermedi come "ping-pong"
    dtype (*current_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_a;
    dtype (*next_buffer)  [MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_b;

    // Alcune variabili...
    int time_length  = TIME_LENGTH;
    int layer, out, b, t, i, k;
    int input_dim, output_dim, kernel_size, dilation;

    int weight_offset = 0;  // dove siamo arrivati in WEIGHTS[]
    int bias_offset   = 0;  // dove siamo arrivati in BIASES[]

    // Ciclo sui layer (NUM_LAYERS è in tcn_network_params.h)
    for (layer = 0; layer < NUM_LAYERS; ++layer) {
        printf("Layer %d...\n", layer);

        input_dim   = (layer == 0) ? INPUT_DIM : HIDDEN_DIMS[layer - 1];
        output_dim  = HIDDEN_DIMS[layer];
        kernel_size = KERNEL_SIZES[layer];
        dilation    = DILATIONS[layer];

        // Convoluzione per canale di uscita
        for (out = 0; out < output_dim; ++out) {
            // Carichiamo i pesi corrispondenti da flash
            if (fill_buffer(&WEIGHTS[weight_offset],
                            buffer_weights,
                            input_dim * kernel_size) != FLASH_OK)
            {
                return EXIT_FAILURE;
            }
            weight_offset += input_dim * kernel_size;

            // Carichiamo il bias
            if (fill_buffer(&BIASES[bias_offset],
                            buffer_bias,
                            1) != FLASH_OK)
            {
                return EXIT_FAILURE;
            }
            bias_offset += 1;

            // Esegui la convolution
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (t = 0; t < time_length; ++t) {
                    dtype result = 0;
                    // cicli su input_dim e kernel_size
                    for (i = 0; i < input_dim; ++i) {
                        for (k = 0; k < kernel_size; ++k) {
                            int index = t - (kernel_size - 1 - k) * dilation;
                            if (index >= 0 && index < time_length) {
                                dtype input_value;
                                if (layer == 0) {
                                    // se layer 0, prendi da REF_INPUT
                                    input_value = (dtype)(REF_INPUT[b * INPUT_DIM * TIME_LENGTH + i * TIME_LENGTH + index] * SCALE);
                                } else {
                                    input_value = current_buffer[b][i][index];
                                }
                                dtype weight = buffer_weights[i*kernel_size + k];

                                result += (input_value * weight) / SCALE;
                            }
                        }
                    }
                    // Aggiungi bias
                    result += buffer_bias[0];

                    // Salva in next_buffer
                    next_buffer[b][out][t] = result;
                }
            }
        }
        printf("    Convolution done\n");

        // Dequantizzi in float
        dequantize_intermediate_buffer(next_buffer, dq_buffer, output_dim, time_length);
        printf("    Dequantized\n");

        // Se c'è la ReLU
        if (RELU_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        if (dq_buffer[b][out][t] < 0) {
                            dq_buffer[b][out][t] = 0.0f;
                        }
                    }
                }
            }
            printf("    ReLU applied\n");
        }

        // Se c'è il maxpool
        if (MAXPOOL_FLAGS[layer]) {
            time_length = time_length / 2; // Dividi per 2
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        float a_ = dq_buffer[b][out][2*t];
                        float b_ = dq_buffer[b][out][2*t+1];
                        dq_buffer[b][out][t] = (a_ > b_) ? a_ : b_;
                    }
                }
            }
            printf("    MaxPool applied\n");
        }

        // Se c'è Global Average Pooling
        if (GAP_FLAGS[layer]) {
            float sum;
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    sum = 0.0f;
                    for (t = 0; t < time_length; ++t) {
                        sum += dq_buffer[b][out][t];
                    }
                    dq_buffer[b][out][0] = sum / time_length;
                }
            }
            time_length = 1;
            printf("    GAP applied\n");
        }

        // Se c'è Softmax
        if (SOFTMAX_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (t = 0; t < time_length; ++t) {
                    float sum_ = 0.0f;
                    float tmp[MAX_HIDDEN_DIM];
                    for (out = 0; out < output_dim; ++out) {
                        tmp[out] = expf(dq_buffer[b][out][t]);
                        sum_ += tmp[out];
                    }
                    for (out = 0; out < output_dim; ++out) {
                        dq_buffer[b][out][t] = tmp[out] / sum_;
                    }
                }
            }
            printf("    Softmax applied\n");
        }

        // Ri-quantizzi dq_buffer -> current_buffer
        requantize_intermediate_buffer(dq_buffer, current_buffer, output_dim, time_length);
        printf("    Requantized\n");
    }

    // L'uscita finale è in current_buffer
    // Dequantizzi su output float [BATCH_SIZE][NUM_CLASSES]
    for (int b_ = 0; b_ < BATCH_SIZE; ++b_) {
        for (int out_ = 0; out_ < NUM_CLASSES; ++out_) {
            output[b_][out_] = (float)current_buffer[b_][out_][0] / SCALE;
        }
    }

    return 0;
}

#endif // MAIN_H_
