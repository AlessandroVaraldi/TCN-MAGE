// Copyright 2024 EPFL
// Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
//
// File: sw/applications/example_data_processing_from_flash/main.h
// Author:  Francesco Poluzzi
// Date: 29/07/2024

#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "x-heep.h"
#include "w25q128jw.h"
#include "core_v_mini_mcu.h"

// === Network Parameters ===
#include "tcn_network_params.h"
#define BATCH_SIZE  1

// === Precision Types ===
#define PRECISION_FLOAT32 1
#define PRECISION_INT8    2
#define PRECISION_INT16   3
#define PRECISION_INT32   4

#define PRECISION   PRECISION_FLOAT32

// === Macros ===
#define SCALE ((PRECISION == PRECISION_FLOAT32) ? 1 : \
              (PRECISION == PRECISION_INT8)   ? (1 << FIXED_POINT >> 2) : \
              (PRECISION == PRECISION_INT16)  ? (1 << FIXED_POINT >> 1) : \
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

// === Global Variables ===
dtype buffer_weights[MAX_HIDDEN_DIM * MAX_HIDDEN_DIM * 3] = {0};
dtype buffer_bias[MAX_HIDDEN_DIM] = {0};
dtype intermediate_a[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH] = {0};
dtype intermediate_b[BATCH_SIZE][MAX_HIDDEN_DIM][TIME_LENGTH] = {0};

// === Function Prototypes ===
int inference(dtype (*input)[INPUT_DIM][TIME_LENGTH], dtype (*output)[NUM_CLASSES]);
int compare_output(const dtype (*output)[NUM_CLASSES]);
int initialize_input(dtype (*input)[INPUT_DIM][TIME_LENGTH]);

static void printFloat(float number, int decimalPlaces);
static void printIntegerPart(float x);

w25q_error_codes_t fill_buffer(float *source, float *buffer, int len);
uint32_t heep_get_flash_address_offset(uint32_t* data_address_lma);

// === Function Implementations ===
int inference(dtype (*input)[INPUT_DIM][TIME_LENGTH], dtype (*output)[NUM_CLASSES])
{
    // current_buffer e next_buffer puntano ai buffer intermedi passati da main
    dtype (*current_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH] = intermediate_a;
    dtype (*next_buffer)[MAX_HIDDEN_DIM][TIME_LENGTH]    = intermediate_b;

    int weight_offset = 0;
    int bias_offset   = 0;

    int num_layers   = NUM_LAYERS;
    int time_length  = TIME_LENGTH;

    // Buffer temporaneo per lo swap
    dtype (*temp)[MAX_HIDDEN_DIM][TIME_LENGTH];

    dtype result, input_value, max_val, sum;
    int layer, b, t, out, i, k, index;

    int input_dim, output_dim, kernel_size, dilation;

    for (layer = 0; layer < num_layers; ++layer) {
        input_dim  = (layer == 0) ? INPUT_DIM : HIDDEN_DIMS[layer - 1];
        output_dim = HIDDEN_DIMS[layer];
        kernel_size = KERNEL_SIZES[layer];
        dilation = DILATIONS[layer];

        if(fill_buffer(&WEIGHTS[weight_offset], buffer_weights, input_dim * output_dim * kernel_size)!=FLASH_OK){
            return EXIT_FAILURE;
        }

        if(fill_buffer(&BIASES[bias_offset], buffer_bias, output_dim)!=FLASH_OK){
            return EXIT_FAILURE;
        }
        
        // --- Apply causal convolution ---
        for (b = 0; b < BATCH_SIZE; ++b) {
            for (t = 0; t < time_length; ++t) {
                for (out = 0; out < output_dim; ++out) {
                    result = 0;
                    i = 0;
                    while (i < input_dim) {
                        for (k = 0; k < kernel_size; ++k) {
                            index = t - (kernel_size - 1 - k) * dilation;
                            if (index >= 0 && index < time_length) {
                                // Se layer == 0 prendiamo direttamente l'input
                                // Altrimenti prendiamo dal buffer intermedio
                                input_value = (layer == 0)
                                              ? input[b][i][index]
                                              : current_buffer[b][i][index];
                                result += ( buffer_weights[out * input_dim * kernel_size 
                                                            + i * kernel_size + k] 
                                            * input_value ) / SCALE;
                            }
                        }
                        ++i;
                    }
                    result += buffer_bias[out];
                    next_buffer[b][out][t] = result;
                }
            }
        }

        // Aggiornamento degli offset in flash
        weight_offset += input_dim * output_dim * kernel_size;
        bias_offset   += output_dim;

        // --- Applica ReLU se richiesto ---
        if (RELU_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        next_buffer[b][out][t] = (next_buffer[b][out][t] > 0)
                                                  ? next_buffer[b][out][t]
                                                  : 0;
                    }
                }
            }
        }

        // --- Applica MaxPool se richiesto ---
        if (MAXPOOL_FLAGS[layer]) {
            time_length = time_length / 2;
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    for (t = 0; t < time_length; ++t) {
                        max_val = next_buffer[b][out][2 * t];
                        if (2 * t + 1 < TIME_LENGTH) {
                            max_val = MAX(max_val, next_buffer[b][out][2 * t + 1]);
                        }
                        next_buffer[b][out][t] = max_val;
                    }
                }
            }
        }

        // --- Applica GAP se richiesto ---
        if (GAP_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (out = 0; out < output_dim; ++out) {
                    dtype sum = 0;
                    for (t = 0; t < time_length; ++t) {
                        sum += next_buffer[b][out][t];
                    }
                    next_buffer[b][out][0] = sum / time_length;
                }
            }
        }

        // --- Applica Softmax se richiesto ---
        if (SOFTMAX_FLAGS[layer]) {
            for (b = 0; b < BATCH_SIZE; ++b) {
                for (t = 0; t < time_length; ++t) {
                    sum = 0;
                    for (out = 0; out < output_dim; ++out) {
                        next_buffer[b][out][t] = exp(next_buffer[b][out][t]);
                        sum += next_buffer[b][out][t];
                    }
                    for (int out2 = 0; out2 < output_dim; ++out2) {
                        next_buffer[b][out2][t] /= sum;
                    }
                }
            }
        }

        // --- Swap dei buffer per il prossimo layer ---
        if (layer < num_layers - 1) {
            temp = current_buffer;
            current_buffer = next_buffer;
            next_buffer = temp;
        }
    }

    // Alla fine, scriviamo il risultato finale nell'output
    for (b = 0; b < BATCH_SIZE; ++b) {
        for (out = 0; out < NUM_CLASSES; ++out) {
            output[b][out] = next_buffer[b][out][0];
        }
    }
    return 0;
}

int load_weights_from_flash(uint32_t offset, dtype* buffer, uint32_t size) {
    uint32_t address_offset = heep_get_flash_address_offset((uint32_t*)WEIGHTS) + offset * sizeof(dtype);
    if (w25q128jw_read_standard(address_offset, (uint8_t*)buffer, size * sizeof(dtype)) != 0) {
        return EXIT_FAILURE;
    }
    return 0;
}

int initialize_input(dtype (*input)[INPUT_DIM][TIME_LENGTH]) {
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < INPUT_DIM; ++j) {
            for (int k = 0; k < TIME_LENGTH; ++k) {
                input[i][j][k] = REF_INPUT[i * INPUT_DIM * TIME_LENGTH + j * TIME_LENGTH + k];
            }
        }
    }
    return 0;
}

static void printIntegerPart(float x) {
    // Caso base: se x < 10 vuol dire che è una singola cifra
    if (x < 10.0f) {
        // Aggiungiamo '0' alla cifra "float" e la passiamo a putchar.
        // Attenzione: '0' è un int, digit è float, quindi c'è comunque
        // un'operazione int+float -> float e poi passaggio implicito a int.
        putchar('0' + x);
        return;
    }
    // Altrimenti scorporiamo l'ultima cifra
    float leading = floorf(x / 10.0f);
    float digit   = fmodf(x, 10.0f);

    // Stampa prima la parte "leading" (tutte le cifre meno l'ultima)
    printIntegerPart(leading);

    // Stampa l'ultima cifra
    putchar('0' + digit);
}

// Stampa un float con un certo numero di cifre decimali, senza usare cast espliciti
static void printFloat(float number, int decimalPlaces) {
    // Gestione del segno

    if (number < 0.0f) {
        putchar('-');
        number = -number;
    }

    // Spezziamo il numero in parte intera e parte frazionaria
    float intPart;
    float fracPart = modff(number, &intPart);

    // Stampa la parte intera
    if (intPart == 0.0f) {
        // Se la parte intera è 0, stampiamo direttamente '0'
        putchar('0');
    } else {
        printIntegerPart(intPart);
    }

    // Stampa il punto decimale
    putchar('.');

    // Stampa la parte decimale
    for (int i = 0; i < decimalPlaces; i++) {
        fracPart = fracPart * 10.0f;
        float digit = floorf(fracPart);
        putchar('0' + digit);
        fracPart = fracPart - digit;
    }
}

w25q_error_codes_t fill_buffer(float *source, float *buffer, int len){
    uint32_t source_flash = heep_get_flash_address_offset((uint32_t*)source);
    w25q_error_codes_t status = w25q128jw_read_standard(source_flash, buffer, (uint32_t) len*4);
    return status;
}

#endif // DATA_H_
