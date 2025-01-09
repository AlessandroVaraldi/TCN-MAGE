#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// Define precision types
#define PRECISION_FLOAT32 1
#define PRECISION_INT8 2
#define PRECISION_INT16 3
#define PRECISION_INT32 4

// === Parametri della rete definiti con #define ===
#define INPUT_DIM 4
#define OUTPUT_DIM 1
#define HIDDEN_DIM 2
#define NUM_LAYERS 3
#define KERNEL_SIZE 3
#define BATCH_SIZE 1
#define TIME_LENGTH 8
#define PRECISION PRECISION_FLOAT32

#if defined(PRECISION) && PRECISION == PRECISION_FLOAT32
typedef float dtype;
#elif defined(PRECISION) && PRECISION == PRECISION_INT8
typedef int8_t dtype;
#elif defined(PRECISION) && PRECISION == PRECISION_INT16
typedef int16_t dtype;
#elif defined(PRECISION) && PRECISION == PRECISION_INT32
typedef int32_t dtype;
#else
#error "Tipo di precisione non supportato!"
#endif

// === Strutture dati per pesi e bias ===
static dtype i_weights[HIDDEN_DIM][INPUT_DIM][KERNEL_SIZE];
static dtype h_weights[NUM_LAYERS-2][HIDDEN_DIM][HIDDEN_DIM][KERNEL_SIZE];
static dtype o_weights[OUTPUT_DIM][HIDDEN_DIM][KERNEL_SIZE];
static dtype biases[NUM_LAYERS-1][HIDDEN_DIM];
static dtype o_bias[OUTPUT_DIM];

// === Buffer statici per input, output e intermedi ===
static dtype input[BATCH_SIZE][INPUT_DIM][TIME_LENGTH];
static dtype output[BATCH_SIZE][OUTPUT_DIM];
static dtype intermediate[BATCH_SIZE][HIDDEN_DIM][TIME_LENGTH];

static int scale;

// Helper per ottenere la stringa di precisione
const char* get_precision_folder() {
    if (PRECISION == PRECISION_FLOAT32) {
        scale = 1;
        return "float32";
    } else if (PRECISION == PRECISION_INT8) {
        scale = pow(2, 4);
        return "int8";
    } else if (PRECISION == PRECISION_INT16) {
        scale = pow(2, 8);
        return "int16";
    } else if (PRECISION == PRECISION_INT32) {
        scale = pow(2, 16);
        return "int32";
    } else {
        scale = 1;
        return "unknown";
    }
}

// Funzione per caricare pesi da file
void load_weights(const char *path) {
    FILE *file;
    char filename[256];
    const char *precision_folder = get_precision_folder();

    snprintf(filename, sizeof(filename), "%s/layer0/weights/%s/data.bin", path, precision_folder);
    file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file dei pesi");
        exit(EXIT_FAILURE);
    }
    fread(i_weights, sizeof(dtype), HIDDEN_DIM * INPUT_DIM * KERNEL_SIZE, file);
    fclose(file);

    snprintf(filename, sizeof(filename), "%s/layer0/biases/%s/data.bin", path, precision_folder);
    file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file dei bias");
        exit(EXIT_FAILURE);
    }
    fread(biases[0], sizeof(dtype), HIDDEN_DIM, file);
    fclose(file);

    for (int layer = 1; layer < NUM_LAYERS-1; ++layer) {
        snprintf(filename, sizeof(filename), "%s/layer%d/weights/%s/data.bin", path, layer, precision_folder);
        file = fopen(filename, "rb");
        if (!file) {
            perror("Errore nell'aprire il file dei pesi");
            exit(EXIT_FAILURE);
        }
        fread(h_weights[layer-1], sizeof(dtype), HIDDEN_DIM * HIDDEN_DIM * KERNEL_SIZE, file);
        fclose(file);

        snprintf(filename, sizeof(filename), "%s/layer%d/weights/%s/data.bin", path, layer, precision_folder);
        file = fopen(filename, "rb");
        if (!file) {
            perror("Errore nell'aprire il file dei bias");
            exit(EXIT_FAILURE);
        }
        fread(biases[layer], sizeof(dtype), HIDDEN_DIM, file);
        fclose(file);
    }

    snprintf(filename, sizeof(filename), "%s/layer%d/weights/%s/data.bin", path, NUM_LAYERS-1, precision_folder);
    file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file dei pesi");
        exit(EXIT_FAILURE);
    }
    fread(o_weights, sizeof(dtype), HIDDEN_DIM * OUTPUT_DIM * KERNEL_SIZE, file);
    fclose(file);

    snprintf(filename, sizeof(filename), "%s/layer%d/biases/%s/data.bin", path, NUM_LAYERS-1, precision_folder);
    file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file dei bias");
        exit(EXIT_FAILURE);
    }
    fread(o_bias, sizeof(dtype), OUTPUT_DIM, file);
    fclose(file);
}

// Funzione per caricare input da file
void load_input(const char *path) {
    char filename[256];
    const char *precision_folder = get_precision_folder();
    snprintf(filename, sizeof(filename), "%s/inputs/%s/data.bin", path, precision_folder);
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file di input");
        exit(EXIT_FAILURE);
    }
    fread(input, sizeof(dtype), BATCH_SIZE * INPUT_DIM * TIME_LENGTH, file);
    fclose(file);
}

// Funzione per eseguire l'inferenza
void inference() {
    // Loop through layers
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int dilation = (layer == 0) ? 1 : (1 << (layer - 1)); // Compute dilation as 2^(layer-1) for hidden layers
        int kernel_half = KERNEL_SIZE / 2; // Kernel size helper for offset calculation

        dtype (*current_weights)[HIDDEN_DIM][KERNEL_SIZE];
        dtype *current_bias;

        if (layer == 0) {
            current_weights = (dtype (*)[HIDDEN_DIM][KERNEL_SIZE])i_weights;
            current_bias = biases[0];
        } else if (layer == NUM_LAYERS - 1) {
            current_weights = (dtype (*)[HIDDEN_DIM][KERNEL_SIZE])o_weights;
            current_bias = o_bias;
        } else {
            current_weights = (dtype (*)[HIDDEN_DIM][KERNEL_SIZE])h_weights[layer - 1];
            current_bias = biases[layer - 1];
        }

        // Loop through the batch and time steps
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < TIME_LENGTH; time++) {
                // Output buffer for this layer
                dtype *output_ptr;
                if (layer == NUM_LAYERS - 1) {
                    output_ptr = &output[batch][0];
                } else {
                    output_ptr = &intermediate[batch][0][time];
                }

                // Initialize output with bias
                for (int o = 0; o < HIDDEN_DIM; o++) {
                    output_ptr[o] = current_bias[o] * scale;
                }

                // Apply convolution
                for (int o = 0; o < HIDDEN_DIM; o++) {
                    for (int i = 0; i < (layer == 0 ? INPUT_DIM : HIDDEN_DIM); i++) {
                        for (int k = 0; k < KERNEL_SIZE; k++) {
                            int input_time = time - k * dilation;
                            if (input_time >= 0) {
                                dtype input_value;
                                if (layer == 0) {
                                    input_value = input[batch][i][input_time];
                                } else {
                                    input_value = intermediate[batch][i][input_time];
                                }
                                output_ptr[o] += current_weights[o][i][k] * input_value;
                            }
                        }
                    }
                }

                // ReLU activation, except for the final layer
                if (layer != NUM_LAYERS - 1) {
                    for (int o = 0; o < HIDDEN_DIM; o++) {
                        if (output_ptr[o] < 0) {
                            output_ptr[o] = 0;
                        }
                    }
                }
            }
        }
    }
}

// Funzione per confrontare output
void compare_output(const char *path) {
    char filename[256];
    const char *precision_folder = get_precision_folder();
    snprintf(filename, sizeof(filename), "%s/outputs/%s/data.bin", path, precision_folder);
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Errore nell'aprire il file di output di riferimento");
        exit(EXIT_FAILURE);
    }

    dtype reference_output[BATCH_SIZE][OUTPUT_DIM];
    fread(reference_output, sizeof(dtype), BATCH_SIZE * OUTPUT_DIM, file);
    fclose(file);

    float float_reference_output[BATCH_SIZE][OUTPUT_DIM];

    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < OUTPUT_DIM; ++o) {
            float_reference_output[b][o] = (float)reference_output[b][o]/scale;
            printf("Output C: %f, Output Python: %f\n", (float)output[b][o], float_reference_output[b][o]);
            if (fabs((float)output[b][o] - float_reference_output[b][o]) > 1e-4) {
                printf("Discrepanza trovata!\n");
                return;
            }
        }
    }
    printf("I risultati coincidono!\n");
}

int main() {
    const char *data_path = "../../Python/data";

    printf("Caricamento dei pesi...\n");
    load_weights(data_path);

    printf("Caricamento dell'input...\n");
    load_input(data_path);

    printf("Esecuzione dell'inferenza...\n");
    inference();

    printf("Confronto con output di riferimento...\n");
    compare_output(data_path);

    return 0;
}
