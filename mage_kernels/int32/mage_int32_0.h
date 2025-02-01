#ifndef INT32_MAGE_LAYER_0_H
#define INT32_MAGE_LAYER_0_H

#include "w25q128jw.h"
#include "mage_cgra.h"

#include "tcn_network_params.h"
#include "int32/weights_bias_int32.h"

#include "mage_dma_int32.h"


/**
 * @brief Configure Mage for the dilated conv1d of layer 0.
 */

/*
    INPUT_DIM = 4
    TIME_LENGTH = 128
    KERNEL_SIZE = 3
    DILATION = 1
    OUTPUT_DIM = 16
*/

void mage_tcn_l0_tile();

void mage_tcn_l0(uint32_t * input_start_addr, uint32_t * outputs_start_addr, int time_length, int kernel_size, int input_dim, int output_dim);

void mage_tcn_l0_tile(){
    //Mage general configuration bits
    uint32_t snt = 0xf;
    uint32_t accMode = 0x20;
    uint32_t blocksize = 0x1;
    uint32_t pea_snt = 0xffffffff;
    mage_set_snt(snt); 
    mage_set_acc(acc_mode); 
    mage_set_address_map_blocksize(blocksize); 
    mage_set_pea_control_snt(pea_snt);

    //Mage Hardware Loops configuration bits
    uint32_t ilb[4] = {0x0, 0x0, 0x0, 0x0};
    uint32_t flb[4] = {0x3, 0x2, 0x80, 0x10};
    uint32_t li[4] = {0x1, 0x1, 0x1, 0x1};
    mage_set_ilb(ilb); 
    mage_set_flb(flb); 
    mage_set_li(li);

    //Mage prolog-kernel-epilog (PKE) values
    uint32_t ii = 0x0;
    uint32_t p = 0x0;
    uint32_t k = 0x0;
    uint32_t e = 0x0;
    uint32_t lenDfg = 0xa;
    mage_set_pke(p, k, e, lenDfg)
    mage_set_ii(ii);

    //Mage Strides configuration bits
    uint32_t strides[8] = {0x18001, 0x18001, 0x0, 0x0, 0x301, 0x301, 0x80010000, 0x0};
    mage_set_strides(strides);
    mage_set_pe_cfg(0x131,0,0,0);
    mage_set_pe_cfg(0x142,0,2,0);
    mage_set_pe_cfg(0x857,1,0,0);
    mage_set_pe_cfg(0x9a,1,1,0);
    mage_set_pe_cfg(0x857,1,2,0);
    mage_set_pe_cfg(0x1070,2,1,0);

    uint32_t loadStreamConfig[1] = {0x22};
    mage_set_load_stream(loadStreamConfig);

    uint32_t storeStreamConfig[1] = {0x0};
    mage_set_store_stream(storeStreamConfig);

    uint32_t selOutPeaConfig[2] = {0x2800, 0x0};
    mage_set_sel_out_pea(selOutPeaConfig);

    mage_set_age_cfg(0x1004800,0,0,0);
    mage_set_age_cfg(0x1804840,0,1,0);
    mage_set_age_cfg(0x1004800,2,0,0);
    mage_set_age_cfg(0x1034840,2,1,0);
    mage_set_age_cfg(0x100a006,3,0,0);

    mage_set_iv_constraints_reg(0x80,3,0);

}

void mage_l0(uint32_t * input_start_addr, uint32_t * outputs_start_addr, int time_length, int kernel_size, int input_dim, int output_dim){

    /*  
        Number of weights = 4*3*16=192 which fits in the Mage memory space dedicated to weights
    */
    if (dma_int32_trans_weights_from_flash(MAGE_WEIGHTS_START_ADDR, &WEIGHTS[weight_offset], output_dim * input_dim * kernel_size) != FLASH_OK)
    {return EXIT_FAILURE;

    /* 
        Number of inputs = 128*4=512 which fits in the Mage memory space dedicated to inputs
    */
    dma_int32_trans_inputs(input_start_addr, MAGE_INPUTS_START_ADDR, time_length, input_dim);

    /*
        Mage Layer 0
    */
    mage_tcn_l0();

    /*
        Number of outputs = 128*16=2048 which fits in the Mage memory space dedicated to outputs
    */
    dma_int32_trans_outputs(MAGE_OUTPUTS_START_ADDR, outputs_start_addr, output_dim, time_length);

}

#endif // INT32_MAGE_LAYER_0_H
