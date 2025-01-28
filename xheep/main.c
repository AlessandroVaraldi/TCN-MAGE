#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "main.h"

#include "x-heep.h"
#include "w25q128jw.h"
#include "dma_sdk.h"

#define FS_INITIAL 0x01

 /* By default, printfs are activated for FPGA and disabled for simulation. */
#define PRINTF_IN_FPGA  1
#define PRINTF_IN_SIM   1
#if TARGET_SIM && PRINTF_IN_SIM
        #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#elif PRINTF_IN_FPGA && !TARGET_SIM
    #define PRINTF(fmt, ...)    printf(fmt, ## __VA_ARGS__)
#else
    #define PRINTF(...)
#endif

float input[BATCH_SIZE][INPUT_DIM][TIME_LENGTH] = {0};
float output[BATCH_SIZE][NUM_CLASSES] = {0};

// === Function Prototypes ===
int initialize_input(float (*input)[INPUT_DIM][TIME_LENGTH]);
int quantize_input(float (*input)[INPUT_DIM][TIME_LENGTH], dtype (*q_input)[INPUT_DIM][TIME_LENGTH]);
int inference(float (*q_input)[INPUT_DIM][TIME_LENGTH], float (*q_output)[NUM_CLASSES]);
int dequantize_output(dtype (*q_output)[NUM_CLASSES], float (*output)[NUM_CLASSES]);
int compare_output(const float (*output)[NUM_CLASSES]);

int main(int argc, char *argv[]) {
#ifndef FLASH_LOAD
    PRINTF("This application is meant to run with the FLASH_LOAD linker script\n");
    return EXIT_SUCCESS;
#else

    //enable FP operations
    CSR_SET_BITS(CSR_REG_MSTATUS, (FS_INITIAL << 13));

    printf("Inizializzazione della periferica di controllo...\n");

    soc_ctrl_t soc_ctrl;
    soc_ctrl.base_addr = mmio_region_from_addr((uintptr_t)SOC_CTRL_START_ADDRESS);

    #ifdef TARGET_SIM
        PRINTF("This application is meant to run on FPGA only\n");
        return EXIT_SUCCESS;
    #endif

    if ( get_spi_flash_mode(&soc_ctrl) == SOC_CTRL_SPI_FLASH_MODE_SPIMEMIO ) {
        PRINTF("This application cannot work with the memory mapped SPI FLASH"
            "module - do not use the FLASH_EXEC linker script for this application\n");
        return EXIT_SUCCESS;
    }

    // Pick the correct spi device based on simulation type
    spi_host_t* spi = spi_flash;

    // Init SPI host and SPI<->Flash bridge parameters
    if (w25q128jw_init(spi) != FLASH_OK){
        PRINTF("Error initializing SPI flash\n");
        return EXIT_FAILURE;
    }

    PRINTF("Inizializzazione dell'input...\n");
    if (initialize_input(input) != EXIT_SUCCESS) {
        PRINTF("Errore inizializzazione input\n");
        return EXIT_FAILURE;
    }

    PRINTF("Esecuzione dell'inferenza...\n");
    if (inference(input, output) != EXIT_SUCCESS) {
        PRINTF("Errore inferenza\n");
        return EXIT_FAILURE;
    }

    PRINTF("Confronto con output di riferimento...\n");
    compare_output(output);

    return EXIT_SUCCESS;

#endif
}

int compare_output(const float (*output)[NUM_CLASSES]) {
    float reference_output[BATCH_SIZE][NUM_CLASSES];
    float current_output;
    float current_reference_output;

    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            reference_output[i][j] = REF_OUTPUT[i * NUM_CLASSES + j];
        }
    }

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < NUM_CLASSES; ++o) {
            current_output = output[b][o];
            current_reference_output = reference_output[b][o];

            PRINTF("Output Python: ");
            printFloat((float)current_reference_output, 6);
            PRINTF(", Output C: ");
            printFloat((float)current_output, 6);
            PRINTF(", ");
            if (fabsf((float)current_output - (float)current_reference_output) > 1e-3) {
                PRINTF("Errore: i valori non corrispondono!\n");
            }
            else {
                PRINTF("Valori corrispondenti.\n");
            }
        }
    }
    return 0;
}
