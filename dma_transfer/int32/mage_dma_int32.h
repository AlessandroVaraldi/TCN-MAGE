#ifndef MAGE_DMA_INT32_H
#define MAGE_DMA_INT32_H

#include "dma.h"
#include "mage_cgra.h"
#include "w25q128jw.h"

void dma_int32_trans_inputs(uint32_t *src_ptr, uint32_t *dst_ptr, uint32_t time_length, uint32_t input_dim);

void dma_int32_trans_weights_from_flash(uint32_t *src_ptr, uint32_t *dst_ptr, uint32_t weight_size, uint32_t kernel_size);

void dma_int32_trans_inputs(uint32_t *src_ptr, uint32_t *dst_ptr, uint32_t time_length, uint32_t input_dim){
    //dma_init(NULL);

    dma* dma_peri;
    dma_peri = dma_peri(0);

    // Input Transfer for even input channels

    // Sign extension
    dma_peri -> SIGN_EXT = 1;
    // mode to read from flash
    dma_peri -> MODE = DMA_TRANS_MODE_SINGLE;
    // dimensionality of the transfer
    dma_peri -> DIM_CONFIG = DMA_DIM_CONF_2D;
    // DMA interrupt enable
    dma_peri -> INTERRUPT_EN = 1;
    // SLOT?
    dma_peri -> SLOT = 0;
    // Source pointer is main memory
    dma_peri -> SRC_PTR = src_ptr;
    // Destination pointer Mage
    dma_peri -> DST_PTR = dst_ptr;
    // Source data type is WORD
    dma_peri -> SRC_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Destination data type is WORD
    dma_peri -> DST_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Increment source pointer
    dma_peri -> SRC_PTR_INC_D1 = 1;
    dma_peri -> SRC_PTR_INC_D2 = time_length;
    // Increment destination pointer d1
    dma_peri -> DST_PTR_INC_D1 = 1;
    dma_peri -> DST_PTR_INC_D2 = 1;
    // Size of the transfer
    dma_peri -> SIZE_D1 = time_length*(input_dim/2);
    
    protected_wait_for_dma_interrupt();

    // Input Transfer for even input channels

    // Sign extension
    dma_peri -> SIGN_EXT = 1;
    // mode to read
    dma_peri -> MODE = DMA_TRANS_MODE_SINGLE;
    // dimensionality of the transfer
    dma_peri -> DIM_CONFIG = DMA_DIM_CONF_2D;
    // DMA interrupt enable
    dma_peri -> INTERRUPT_EN = 1;
    // SLOT?
    dma_peri -> SLOT = 0;
    // Source pointer is flash
    dma_peri -> SRC_PTR = src_ptr + time_length*(input_dim/2);
    // Destination pointer Mage
    dma_peri -> DST_PTR = ;
    // Source data type is WORD
    dma_peri -> SRC_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Destination data type is WORD
    dma_peri -> DST_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Increment source pointer
    dma_peri -> SRC_PTR_INC_D1 = 1;
    dma_peri -> SRC_PTR_INC_D2 = time_length;
    // Increment destination pointer d1
    dma_peri -> DST_PTR_INC_D1 = 1;
    dma_peri -> DST_PTR_INC_D2 = 1;
    // Size of the transfer
    dma_peri -> SIZE_D1 = time_length*(input_dim/2);
}

// to be modifed
void dma_int32_trans_weights_from_flash(uint32_t *src_ptr, uint32_t *dst_ptr, uint32_t weight_size, uint32_t kernel_size){
    //dma_init(NULL);

    // The DMA will wait for the SPI HOST/FLASH RX FIFO valid signal
    #ifndef USE_SPI_FLASH
    uint8_t slot = DMA_TRIG_SLOT_SPI_RX;
    #else
    uint8_t slot = DMA_TRIG_SLOT_SPI_FLASH_RX;
    #endif

    dma* dma_peri;
    dma_peri = dma_peri(0);

    // Weigth Transfer for even input channels

    // Sign extension
    dma_peri -> SIGN_EXT = 1;
    // mode to read from flash
    dma_peri -> MODE = DMA_TRANS_MODE_SUBADDRESS;
    // dimensionality of the transfer
    dma_peri -> DIM_CONFIG = DMA_DIM_CONF_1D;
    // DMA interrupt enable
    dma_peri -> INTERRUPT_EN = 1;
    // SLOT?
    dma_peri -> SLOT = slot;
    // Source pointer is flash
    dma_peri -> SRC_PTR = (uint8_t*) (src_ptr);
    // Destination pointer Mage
    dma_peri -> DST_PTR = ;
    // Source data type is WORD
    dma_peri -> SRC_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Destination data type is WORD
    dma_peri -> DST_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Increment source pointer by 0
    dma_peri -> SRC_PTR_INC_D1 = 0;
    // Increment destination pointer d1
    dma_peri -> DST_PTR_INC_D1 = 1;
    // Size of the transfer
    dma_peri -> SIZE_D1 = time_length*(input_dim/2);
    
    protected_wait_for_dma_interrupt();

    // Input Transfer for even input channels

    // Sign extension
    dma_peri -> SIGN_EXT = 1;
    // mode to read from flash
    dma_peri -> MODE = DMA_TRANS_MODE_SUBADDRESS;
    // dimensionality of the transfer
    dma_peri -> DIM_CONFIG = DMA_DIM_CONF_1D;
    // DMA interrupt enable
    dma_peri -> INTERRUPT_EN = 1;
    // SLOT?
    dma_peri -> SLOT = slot;
    // Source pointer is flash
    dma_peri -> SRC_PTR = (uint8_t*) (src_ptr + time_length*(input_dim/2));
    // Destination pointer Mage
    dma_peri -> DST_PTR = ;
    // Source data type is WORD
    dma_peri -> SRC_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Destination data type is WORD
    dma_peri -> DST_DATA_TYPE = DMA_DATA_TYPE_WORD;
    // Increment source pointer by 0
    dma_peri -> SRC_PTR_INC_D1 = 0;
    // Increment destination pointer d1
    dma_peri -> DST_PTR_INC_D1 = 1;
    // Size of the transfer
    dma_peri -> SIZE_D1 = time_length*(input_dim/2);
}

#endif // INT32_DMA_0_H
