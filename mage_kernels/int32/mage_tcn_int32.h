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

        if (layer > 0)
            input_start_addr = outputs_start_addr;

        for (out = 0; out < output_dim; ++out) {
            if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, &WEIGHTS[weight_offset], input_dim, kernel_size) != FLASH_OK)
            {
                return EXIT_FAILURE;
            }
            
            weight_offset += input_dim * kernel_size;
            
            dma_int32_trans_inputs(input_start_addr, MAGE_INPUTS_START_ADDR, time_length, input_dim);
            input_start_addr += input_dim * time_length;

            if (layer == 0) {
                mage_dil_conv1d_layer_0();
                dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);
            } else if (layer == 1) {
                mage_dil_conv1d_layer_1();
                dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);
            } else if (layer == 2) {
                mage_dil_conv1d_layer_2();
                dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);
            } else if (layer == 3) {
                mage_dil_conv1d_layer_3();
                dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);
            } else if (layer == 4) {
                mage_dil_conv1d_layer_4();
                dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);
            } else if (layer == 5) {
                mage_dil_conv1d_layer_5();
                dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);
            }

            outputs_start_addr += output_dim * time_length;

        }

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