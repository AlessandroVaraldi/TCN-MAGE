#ifndef INT32_MAGE_LAYER_2_H
#define INT32_MAGE_LAYER_2_H

#include "mage_cgra.h"

/*
    Parameters for this layer are:
    
    INPUT_DIM = 32
    TIME_LENGTH = 128
    KERNEL_SIZE = 3
    DILATION = 4
    OUTPUT_DIM = 64
*/

void mage_l2_tile(uint32_t output_ch, uint32_t input_ch);

void mage_l2(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim, int n_pad_elements);

void mage_l2_tile(uint32_t output_ch, uint32_t input_ch){
}

/*
    In this layer, weights, inputs and outputs don't fit in Mage
    For a given batch of outputs, the related weights are transferred to Mage from Flash
    10/10/12 input channels are transferred from Memory to Mage in three tiles to generate one batch of outputs
    Outputs are transferred to Memory once calculated

    Number of inputs = (128+(3-1)*4)*32=4352 which does not fit in the Mage memory space dedicated to inputs            
    so we have to tile the inputs in two parts and transfer them in two parts

    Number of weights = 32*3*64=6144 which does not fit in the Mage memory space dedicated to weights
    so we have to tile the inputs in two parts and transfer them in two parts along with the inputs

    Number of outputs = 128*64=8192 which does not fit in the Mage memory space dedicated to outputs
    so we have to tile the kernel in four parts and transfer the outputs in four parts
*/
void mage_l2(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim, int n_pad_elements){

    uint32_t o_tile_size = 16;
    uint32_t i_tile_size = {11, 11, 10};
    uint32_t weights_tile_size = o_tile_size * input_dim * kernel_size;

    uint32_t num_o_tiles = output_dim / o_tile_size;
    uint32_t num_i_tiles = (uint32_t) (input_dim / i_tile_size[0]);

    uint32_t * curr_input_start_addr = input_start_addr;
    uint32_t * curr_weights_start_addr = weights_start_addr;

    for(int o = 0; o < num_o_tiles; o++){

        // Transfer weights for the current output channel for all input channels 16*3*32=1536
        if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, curr_weights_start_addr, weights_tile_size) != FLASH_OK)
        {return EXIT_FAILURE;}
        curr_weights_start_addr += weights_tile_size;
        
        for(int i = 0; i < num_i_tiles; i++){
                
            // Transfer inputs of tile (o, i)
            dma_int32_trans_inputs(curr_input_start_addr, MAGE_INPUTS_START_ADDR, time_length, i_tile_size[i], n_pad_elements);
            curr_input_start_addr += i_tile_size[i] * time_length;

            mage_l2_tile(o, i);
        
        }

        dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, o_tile_size, time_length);
        outputs_start_addr += o_tile_size * time_length;
        curr_input_start_addr = input_start_addr;
    
    }

}

#endif // INT32_MAGE_LAYER_2_H