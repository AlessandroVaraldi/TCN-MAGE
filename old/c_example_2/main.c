#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "conv1d.h"  // Include il file delle funzioni per la convoluzione
#include "tcn_layer.h"  // Include il file delle funzioni per il layer TCN
#include "tcn_network.h"  // Include il file per il modello TCN

// Funzione per caricare i dati CSV (come definito in precedenza)
void load_csv(const char* filename, float*** data, int* num_rows, int* num_cols);

// Funzione per creare le sequenze temporali (come definito in precedenza)
void create_sequences(float** data, int num_rows, int num_cols, int sequence_length, float*** X, float*** y);

int main() {
    // Variabili per gestire i dati
    float** data = NULL;
    float*** X = NULL;
    float*** y = NULL;
    int num_rows = 0, num_cols = 0;

    // 1. Caricamento del file CSV
    const char* file_path = "data.csv";  // Percorso al file CSV
    load_csv(file_path, &data, &num_rows, &num_cols);

    // 2. Creazione delle sequenze temporali
    int sequence_length = 10;  // Lunghezza delle sequenze temporali
    create_sequences(data, num_rows, num_cols, sequence_length, &X, &y);

    // 3. Definizione dei pesi per i layer (ogni layer avrà un kernel di convoluzione)
    int num_layers = 3;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;

    // Allocazione dei pesi (kernel) per ogni layer
    float*** kernels = (float***)malloc(num_layers * sizeof(float**));
    for (int i = 0; i < num_layers; i++) {
        kernels[i] = (float**)malloc(num_cols * sizeof(float*));  // Numero di feature (colonne)
        for (int j = 0; j < num_cols; j++) {
            kernels[i][j] = (float*)malloc(kernel_size * sizeof(float));
            for (int k = 0; k < kernel_size; k++) {
                kernels[i][j][k] = 0.01f;  // Inizializzazione semplice con piccoli valori
            }
        }
    }

    // 4. Eseguiamo l'inferenza nel modello TCN
    float* output = (float*)malloc(num_cols * sizeof(float));  // Variabile per memorizzare l'output finale

    // Passiamo i dati attraverso il modello TCN
    tcn_network(X, output, kernels, num_rows - sequence_length + 1, num_layers, kernel_size, stride, padding);

    // 5. Visualizza i risultati
    printf("Risultati dell'inferenza:\n");
    for (int i = 0; i < num_cols; i++) {
        printf("Predizione per colonna %d: %f\n", i + 1, output[i]);
    }

    // 6. Pulizia della memoria
    // Dealloca memoria per X, y, e kernels
    for (int i = 0; i < num_rows - sequence_length + 1; i++) {
        for (int j = 0; j < num_cols; j++) {
            free(X[i][j]);
        }
        free(X[i]);
        free(y[i]);
    }
    free(X);
    free(y);

    for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < num_cols; j++) {
            free(kernels[i][j]);
        }
        free(kernels[i]);
    }
    free(kernels);

    free(data);
    free(output);

    return 0;
}

// Funzione per caricare il CSV
void load_csv(const char* filename, float*** data, int* num_rows, int* num_cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Errore nell'aprire il file CSV.\n");
        exit(1);
    }

    char line[1024];
    int row = 0;

    // Conta le righe e le colonne
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        int col = 0;
        while (token) {
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }

    // Torna all'inizio del file
    fseek(file, 0, SEEK_SET);

    // Allocazione dinamica per i dati
    *data = (float**)malloc(row * sizeof(float*));
    for (int i = 0; i < row; i++) {
        (*data)[i] = (float*)malloc(*num_cols * sizeof(float));
    }

    // Legge effettivamente i dati nel file
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        int j = 0;
        while (token) {
            (*data)[i][j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }

    fclose(file);
}

// Funzione per creare le sequenze temporali
void create_sequences(float** data, int num_rows, int num_cols, int sequence_length, float*** X, float*** y) {
    // Calcola il numero di sequenze che possiamo creare
    int num_sequences = num_rows - sequence_length + 1;

    // Allocazione dinamica per X e y
    *X = (float***)malloc(num_sequences * sizeof(float**));  // X è un array di sequenze (3D)
    *y = (float**)malloc(num_sequences * sizeof(float*));    // y è un array di output per ogni sequenza

    // Allocazione per X
    for (int i = 0; i < num_sequences; i++) {
        (*X)[i] = (float**)malloc(num_cols * sizeof(float*));  // Un array di puntatori a float per ogni feature
        for (int j = 0; j < num_cols; j++) {
            (*X)[i][j] = (float*)malloc(sequence_length * sizeof(float));  // Un array di float per ogni sequenza di feature
        }
    }

    // Allocazione per y (output)
    for (int i = 0; i < num_sequences; i++) {
        (*y)[i] = (float*)malloc(num_cols * sizeof(float));  // Un array di float per l'output (tutte le feature)
    }

    // Creazione delle sequenze temporali
    for (int i = 0; i < num_sequences; i++) {
        for (int j = 0; j < num_cols; j++) {
            for (int k = 0; k < sequence_length; k++) {
                (*X)[i][j][k] = data[i + k][j];  // Copia i dati in X[i][j][k]
            }
        }

        // L'output (y) è l'ultima riga della sequenza (l'ultima istanza della sequenza)
        for (int j = 0; j < num_cols; j++) {
            (*y)[i][j] = data[i + sequence_length - 1][j];  // L'output è l'ultima riga della sequenza
        }
    }
}
