#ifndef INT32_MAGE_LAYER_3_H
#define INT32_MAGE_LAYER_3_H

#include "mage_cgra.h"

/**
 * @brief Configure Mage for the dilated conv1d of layer 1.
 */

/*
    INPUT_DIM = 64
    TIME_LENGTH = 128
    KERNEL_SIZE = 3
    DILATION = 8
    OUTPUT_DIM = 128
*/

void mage_l3_tile();
void mage_l3(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim);

void mage_l3_tile(){
}

void mage_l3(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim){
    /* 
        Number of weights = 64*3*128=24576 which does not fit in the Mage memory space dedicated to weights

        Number of inputs = 128*64=8192 which does not fit in the Mage memory space dedicated to inputs
        so we have to tile the inputs in four parts and transfer them in four parts

        Number of outputs = 128*128=16384 which does not fit in the Mage memory space dedicated to outputs
        so we have to tile the kernel in eight parts and transfer the outputs in eight parts
    */
    
    uint32_t o_tile_size = 16;
    uint32_t i_tile_size = 16;
    uint32_t weights_tile_size = i_tile_size * kernel_size;

    uint32_t num_o_tiles = output_dim / o_tile_size;
    uint32_t num_i_tiles = input_dim / i_tile_size;

    uint32_t * curr_input_start_addr = input_start_addr;
    uint32_t * curr_weights_start_addr = weights_start_addr;

    for(int o = 0; o < num_o_tiles; o++){
        for(int i = 0; i < num_i_tiles; i++){
                
            // Transfer weights of tile (o, i)
            if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, curr_weights_start_addr, weights_tile_size) != FLASH_OK)
            {return EXIT_FAILURE;}
            curr_weights_start_addr += weights_tile_size;

            // Transfer inputs of tile (o, i)
            dma_int32_trans_inputs(curr_input_start_addr, MAGE_INPUTS_START_ADDR, time_length, i_tile_size);
            curr_input_start_addr += i_tile_size * time_length;

            mage_l2_tile();
        
        }

        dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, o_tile_size, time_length);
        outputs_start_addr += o_tile_size * time_length;
        curr_input_start_addr = input_start_addr;
        curr_weights_start_addr = weights_start_addr;
    
    }

}


#endif // INT32_MAGE_LAYER_3_H