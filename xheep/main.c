/**
 * @file main.c
 * @brief Header file for example data processing from flash application.
 * 
 * This file contains the necessary includes, definitions, and function 
 * declarations for the example data processing from flash application.
 * 
 * @author Alessandro Varaldi
 * @date 21/01/2025
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "x-heep.h"
#include "w25q128jw.h"
#include "dma_sdk.h"

#include "main.h"

#define FS_INITIAL 0x01

 /* By default, printfs are activated for FPGA and disabled for simulation. */
#define PRINTF_IN_FPGA  1 // Set to 1 to enable printf in FPGA, 0 to disable
#define PRINTF_IN_SIM   0

#if TARGET_SIM && PRINTF_IN_SIM
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif PRINTF_IN_FPGA && !TARGET_SIM
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
    #define PRINTF(...)
#endif

dtype input[BATCH_SIZE][INPUT_DIM][TIME_LENGTH] = {0};
dtype output[BATCH_SIZE][NUM_CLASSES] = {0};
// === Function Prototypes ===
int initialize_input(dtype (*input)[INPUT_DIM][TIME_LENGTH]);
int inference(dtype (*input)[INPUT_DIM][TIME_LENGTH], dtype (*output)[NUM_CLASSES]);
int compare_output(const dtype (*output)[NUM_CLASSES]);

uint32_t heep_get_flash_address_offset(uint32_t* data_address_lma);

int load_weights_from_flash(uint32_t offset, dtype* buffer, uint32_t size);

int main(int argc, char *argv[]) {

    #ifndef FLASH_LOAD
        PRINTF("This application is meant to run with the FLASH_LOAD linker script\n");
        return EXIT_SUCCESS;
    #else

    //enable FP operations
    CSR_SET_BITS(CSR_REG_MSTATUS, (FS_INITIAL << 13));

    printf("Initializing control peripheral...\n");

    soc_ctrl_t soc_ctrl;
    soc_ctrl.base_addr = mmio_region_from_addr((uintptr_t)SOC_CTRL_START_ADDRESS);

    #ifdef TARGET_SIM
        PRINTF("This application is meant to run on FPGA only\n");
        return EXIT_SUCCESS;
    #endif

    if ( get_spi_flash_mode(&soc_ctrl) == SOC_CTRL_SPI_FLASH_MODE_SPIMEMIO ) {
        PRINTF("This application cannot work with the memory mapped SPI FLASH module - do not use the FLASH_EXEC linker script for this application\n");
        return EXIT_SUCCESS;
    }

    // Pick the correct spi device based on simulation type
    spi_host_t* spi = spi_flash;

    // Init SPI host and SPI<->Flash bridge parameters
    if (w25q128jw_init(spi) != FLASH_OK){
        PRINTF("Error initializing SPI flash\n");
        return EXIT_FAILURE;
    }

    PRINTF("Initializing input...\n");
    initialize_input(input);

    PRINTF("Executing inference...\n");
    inference(input, output);

    PRINTF("Comparing with reference output...\n");
    compare_output(output);

    return EXIT_SUCCESS;

#endif
}

/**
    // reference_output as a local automatic array (non-static)
 *
 * @param output The output of the inference to be compared.
 * @return 0 if the outputs match, otherwise an error message is printed.
 */
int compare_output(const dtype (*output)[NUM_CLASSES]) {
    // reference_output come array locale automatico (non statico)
    dtype reference_output[BATCH_SIZE][NUM_CLASSES];
    dtype current_output;
    dtype current_reference_output;

    // Carichiamo l'output di riferimento nello stesso formato
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            reference_output[i][j] = REF_OUTPUT[i * NUM_CLASSES + j];
        }
    }

    // Confronto uno a uno
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < NUM_CLASSES; ++o) {
            current_output = output[b][o];
            current_reference_output = reference_output[b][o];

            if (PRECISION == PRECISION_FLOAT32) {
                float diff = (float)current_output - (float)current_reference_output;
                PRINTF("Output Python: ");
                printFloat((float)current_reference_output, 6);
                PRINTF(", Output C: ");
                printFloat((float)current_output, 6);
                PRINTF(", ");
                if (fabsf(diff) > 1e-3) {
                    PRINTF("Errore: i valori non corrispondono!\n");
                }
                else {
                    PRINTF("Valori corrispondenti.\n");
                }
            }
            else {
                PRINTF("Output Python: %d, Output C: %d, ",
                       current_reference_output, current_output);
                if (current_output != current_reference_output) {
                    PRINTF("Errore: i valori non corrispondono!\n");
                }
                else {
                    PRINTF("Valori corrispondenti.\n");
                }
            }
        }
    }
    return 0;
}

// === Function Implementations ===

