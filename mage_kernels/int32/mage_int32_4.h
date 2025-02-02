#ifndef INT32_MAGE_LAYER_4_H
#define INT32_MAGE_LAYER_4_H

#include "mage_cgra.h"

/*
    Parameters for this layer are:
    
    INPUT_DIM = 128
    TIME_LENGTH = 128
    KERNEL_SIZE = 1
    DILATION = 1
    OUTPUT_DIM = 64
*/

void mage_l4_tile(uint32_t output_ch, uint32_t input_ch);
void mage_l4(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim);

void mage_l4_tile(uint32_t output_ch, uint32_t input_ch){
}

/*
    In this layer, weights, inputs and outputs don't fit in Mage
    16/16/16/16/16/16/16/16 input channels are transferred from Memory to Mage in 8 tiles to generate one batch of outputs
    In those 5 tiles, also weights must be transferred according to the current input and output channels
    Outputs are transferred to Memory once calculated

    Number of weights = 128*3*64=24576 which does not fit in the Mage memory space dedicated to weights

    Number of inputs = 128*128=16384 which does not fit in the Mage memory space dedicated to inputs
    so we have to tile the inputs in eight parts and transfer them in eight parts

    Number of outputs = 128*64=8192 which does not fit in the Mage memory space dedicated to outputs
    so we have to tile the kernel in four parts and transfer the outputs in four parts
*/
void mage_l4(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim){

    uint32_t o_tile_size = 16;
    uint32_t i_tile_size = 16;
    uint32_t weights_tile_size = i_tile_size * kernel_size;

    uint32_t num_o_tiles = output_dim / o_tile_size;
    uint32_t num_i_tiles = input_dim / i_tile_size;

    uint32_t * curr_input_start_addr = input_start_addr;
    uint32_t * curr_weights_start_addr = weights_start_addr;

    for(int o = 0; o < num_o_tiles; o++){
        for(int i = 0; i < num_i_tiles; i++){
                
            // Transfer weights of tile (o, i) at most 16*3*14=672
            if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, curr_weights_start_addr, weights_tile_size) != FLASH_OK)
            {return EXIT_FAILURE;}
            curr_weights_start_addr += weights_tile_size;
            weights_tile_size = i_tile_size * kernel_size;

            // Transfer inputs of tile (o, i) at most 128*14=1792
            dma_int32_trans_inputs(curr_input_start_addr, MAGE_INPUTS_START_ADDR, time_length, i_tile_size, n_pad_elements);
            curr_input_start_addr += i_tile_size * time_length;

            mage_l3_tile(o, i);
        
        }

        // Transfer outputs of tile (o) 128*16=2048
        dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, o_tile_size, time_length);
        outputs_start_addr += o_tile_size * time_length;
        curr_input_start_addr = input_start_addr;
        curr_weights_start_addr = weights_start_addr;
    
    }

}

#endif // INT32_MAGE_LAYER_4_H