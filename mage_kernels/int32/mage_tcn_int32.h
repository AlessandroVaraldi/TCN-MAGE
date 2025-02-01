#ifndef MAGE_TCN_INT32_H
#define MAGE_TCN_INT32_H

#include "mage_cgra.h"
#include "w25q128jw.h"
#include "dma.h"

#include "weights_bias_int32.h"
#include "tcn_network_params.h"
#include "mage_dma_int32.h"
#include "mage_int32_0.h"
#include "mage_int32_1.h"
#include "mage_int32_2.h"
#include "mage_int32_3.h"
#include "mage_int32_4.h"
#include "mage_int32_5.h"

#define MAGE_WEIGHTS_START_ADDR MAGE_BANK_0
#define MAGE_INPUTS_START_ADDR MAGE_BANK_4
#define MAGE_OUTPUTS_START_ADDR MAGE_BANK_6


void mage_tcn_int32();

void mage_tcn_int32(){

    int time_length  = TIME_LENGTH;
    int layer, out, b, t, i, k;
    int input_dim, output_dim, kernel_size, dilation;

    int weight_offset = 0;
    int bias_offset   = 0;

    uint32_t * input_start_addr = REF_INPUT;
    uint32_t * outputs_start_addr;

    for (layer = 0; layer < NUM_LAYERS; ++layer) {
        printf("Layer %d...\n", layer);

        input_dim   = (layer == 0) ? INPUT_DIM : HIDDEN_DIMS[layer - 1];
        output_dim  = HIDDEN_DIMS[layer];
        kernel_size = KERNEL_SIZES[layer];
        dilation    = DILATIONS[l ayer];

        /*
            Mage memory dedicated to inputs = 2048 32-bit words on 2 banks
            Mage memory dedicated to outputs = 2048 32-bit words on 2 banks
            Mage memory dedicated to weights = 2048 32-bit words on 2 banks
        */

        if (layer > 0)
            input_start_addr = outputs_start_addr;

        /*
            one DMA transfer is enough as the max amount of weights is 128*3=384
            which fits in the Mage memory space dedicated to weights
        */
        if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, &WEIGHTS[weight_offset], input_dim, kernel_size) != FLASH_OK)
        {
            return EXIT_FAILURE;
        }
        
        // Move the weight offset to the start address of the next output channel weights
        weight_offset += input_dim * kernel_size;
        
        /*
            Mage memory space dedicated to inputs is 2048 32-bit words on 2 banks
            The maximum amount of inputs to be loaded is 128*128=16384 which does not fit in this space
        */


        printf("    Convolution done\n");

        dequantize_intermediate_buffer(next_buffer, dq_buffer, output_dim, time_length);
        printf("    Dequantized\n");

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

        if (MAXPOOL_FLAGS[layer]) {
            time_length = time_length / 2;
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

        requantize_intermediate_buffer(dq_buffer, current_buffer, output_dim, time_length);
        printf("    Requantized\n");
    }

    for (int b_ = 0; b_ < BATCH_SIZE; ++b_) {
        for (int out_ = 0; out_ < NUM_CLASSES; ++out_) {
            output[b_][out_] = (float)current_buffer[b_][out_][0] / SCALE;
        }
    }

    return 0;

}

#endif // MAGE_TCN_INT32_H