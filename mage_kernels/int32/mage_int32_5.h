#ifndef INT32_MAGE_LAYER_5_H
#define INT32_MAGE_LAYER_5_H

#include "mage_cgra.h"


/*
    Parameters for this layer are:
    
    INPUT_DIM = 64
    TIME_LENGTH = 128
    KERNEL_SIZE = 1
    DILATION = 1
    OUTPUT_DIM = 6
*/

void mage_l5_tile(uint32_t output_ch, uint32_t input_ch);
void mage_l5(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim);

void mage_l5_tile(){
}

/*
    In this layer, inputs don't fit in Mage, while outputs and weights do
    16/16/16/16 input channels are transferred from Memory to Mage in 4 tiles to generate one batch of outputs
    In those 5 tiles, also weights must be transferred according to the current input and output channels
    Outputs are transferred to Memory once calculated

    Number of weights = 64*3*6=1152 which fits in the Mage memory space dedicated to weights

    Number of inputs = 128*64=8192 which does not fit in the Mage memory space dedicated to inputs
    so we have to tile the inputs in four parts and transfer them in four parts

    Number of outputs = 128*6=768 which fits in the Mage memory space dedicated to outputs
*/
void mage_l5(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim){
    
    uint32_t o_tile_size = 6;
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
            dma_int32_trans_inputs(curr_input_start_addr, MAGE_INPUTS_START_ADDR, time_length, i_tile_size, n_pad_elements);
            curr_input_start_addr += i_tile_size * time_length;

            mage_l5_tile(o, i);
        
        }

        dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, o_tile_size, time_length);
        outputs_start_addr += o_tile_size * time_length;
        curr_input_start_addr = input_start_addr;
        curr_weights_start_addr = weights_start_addr;
    
    }

}
#endif // INT32_MAGE_LAYER_5_H