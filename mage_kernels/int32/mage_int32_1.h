#ifndef INT32_MAGE_LAYER_1_H
#define INT32_MAGE_LAYER_1_H

#include "mage_cgra.h"

/**
 * @brief Configure Mage for the dilated conv1d of layer 1.
 */

/*
    INPUT_DIM = 16
    TIME_LENGTH = 128
    KERNEL_SIZE = 3
    DILATION = 2
    OUTPUT_DIM = 32
*/

void mage_l1_tile();
void mage_l1(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim);

void mage_l1_tile(){
}

void mage_l1(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim){
    /* 
        Number of weights = 32*3*16=1535 which fits in the Mage memory space dedicated to weights

        Number of inputs = 128*16=2048 which fits in the Mage memory space dedicated to inputs

        Number of outputs = 128*32=4096 which does not fit in the Mage memory space dedicated to outputs
        so we have to tile the kernel in four parts and transfer the outputs in 2 parts
    */
    
    if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, weights_start_addr, output_dim * input_dim * kernel_size) != FLASH_OK)
    {return EXIT_FAILURE;}

    dma_int32_trans_inputs(input_start_addr, MAGE_INPUTS_START_ADDR, time_length, input_dim);
    
    mage_l1_tile_o0_i0();
    
    dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim/2, time_length);
    outputs_start_addr += (output_dim/2) * time_length;
    
    mage_l1_tile_o1_i0();

    dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim/2, time_length);

}


#endif // INT32_MAGE_LAYER_1_H