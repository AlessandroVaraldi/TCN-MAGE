#ifndef INT32_MAGE_LAYER_1_H
#define INT32_MAGE_LAYER_1_H

#include "mage_cgra.h"

/*
    Parameters for this layer are:
    
    INPUT_DIM = 16
    TIME_LENGTH = 128
    KERNEL_SIZE = 3
    DILATION = 2
    OUTPUT_DIM = 32
*/

void mage_l1_tile(uint32_t output_ch, uint32_t input_ch);

void mage_l1(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim, int n_pad_elements);

void mage_l1_tile(uint32_t output_ch, uint32_t input_ch){
}

/*
    In this layer, weights fit in Mage, while inputs and outputs don't
    All weights are transferred to Mage from Flash
    Half the inputs are transferred from Memory to Mage in two tiles to generate one batch of outputs
    Outputs are transferred to Memory

    Number of weights = 32*3*16=1535 which fits in the Mage memory space dedicated to weights                 

    Number of inputs = (128+(3-1)*2)*16=2112 which does not fit in the Mage memory space dedicated to inputs  

    Number of outputs = 128*32=4096 which does not fit in the Mage memory space dedicated to outputs   
    so we have to tile the kernel in four parts and transfer the outputs in 2 parts
*/
void mage_l1(uint32_t * input_start_addr, uint32_t * outputs_start_addr, uint32_t * weights_start_addr, int time_length, int kernel_size, int input_dim, int output_dim, int n_pad_elements){

    uint32_t o_tile_size = 16;
    uint32_t i_tile_size = 8;
    uint32_t weights_tile_size = output_dim * input_dim * kernel_size;

    uint32_t num_o_tiles = output_dim / o_tile_size;
    uint32_t num_i_tiles = input_dim / i_tile_size;

    uint32_t * curr_input_start_addr = input_start_addr;
    uint32_t * curr_weights_start_addr = weights_start_addr;

    // Transferring all weigths from Flash to Mage
    if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, curr_weights_start_addr, weights_tile_size) != FLASH_OK)
    {return EXIT_FAILURE;}

    for(int o = 0; o < num_o_tiles; o++){ // num_o_tiles = 2
        for(int i = 0; i < num_i_tiles; i++){ // num_i_tiles = 2

            // Transfer inputs of tile (o, i)
            dma_int32_trans_inputs(curr_input_start_addr, MAGE_INPUTS_START_ADDR, time_length, i_tile_size, n_pad_elements);
            curr_input_start_addr += i_tile_size * time_length;

            mage_l1_tile(o, i);
        
        }

        dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, o_tile_size, time_length);
        outputs_start_addr += o_tile_size * time_length;
        curr_input_start_addr = input_start_addr;
    
    }





    
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