#include "causal_conv1d.h"
#include <string.h> // Per l'uso di memset
#include <stdio.h> // Per la stampa di debug
#include <time.h>

/**
 * Esegue la convoluzione causale 1D per un livello (layer) del modello TCN.
 *
 * Parametri:
 *  - layer: puntatore alla struttura CausalConv1dLayer che contiene i parametri (pesi, bias, ecc.).
 *  - input: array di input di dimensioni [batch, in_channels, input_length].
 *           L'ordinamento in memoria è: primo il batch, poi il canale, poi il tempo.
 *  - output: array di output di dimensioni [batch, out_channels, input_length].
 *            Viene scritto da questa funzione.
 *  - batch: numero di sequenze elaborate in parallelo (dimensione della batch).
 *  - in_channels: numero di canali in input a questo layer.
 *  - input_length: lunghezza della sequenza temporale di input.
 *
 * Il layer applica una convoluzione causale con:
 *  - out_channels = layer->out_channels
 *  - kernel_size  = layer->kernel_size
 *  - dilation     = layer->dilation
 *
 * La convoluzione causale assicura che l'output ad un tempo t dipenda solo da istanti t e precedenti,
 * non da valori futuri.
 */

void causal_conv1d_forward(
    const CausalConv1dLayer *layer,
    const float *input,
    float *output,
    int batch,
    int in_channels,
    int input_length,
    char INFO_MODE
) {
    // Azzeriamo l'output prima di iniziare (impostando a zero tutti i valori).
    // Questo perché in seguito aggiungeremo il bias e il contributo dei vari canali e kernel.
    // sizeof(float) * batch * out_channels * input_length = quantità di memoria da azzerare.
    memset(output, 0, sizeof(float) * batch * layer->out_channels * input_length);

    // Estraiamo i parametri del layer per comodità
    int dilation = layer->dilation;
    int ks = layer->kernel_size;
    int oc = layer->out_channels;
    int ic = layer->in_channels; // Deve coincidere con in_channels passato come argomento.
    printf("%c", INFO_MODE);
    if (INFO_MODE) printf("out_channels=%d, in_channels=%d, kernel_size=%d, dilation=%d\n", oc, ic, ks, dilation);

    // Misurazione del tempo di esecuzione del batch
    clock_t start_time = clock();

    // Iteriamo su ogni sequenza del batch
    for (int b = 0; b < batch; b++) {
        // Iteriamo sui canali di output
        for (int oc_i = 0; oc_i < oc; oc_i++) {
            // Prima di tutto aggiungiamo il bias a tutto l'output di questo canale.
            // Poiché l'output atteso è [batch, out_channels, input_length],
            // la posizione in memory per un dato b, oc_i, t è:
            // output[b * oc * input_length + oc_i * input_length + t].
            for (int t = 0; t < input_length; t++) {
                output[b * oc * input_length + oc_i * input_length + t] = layer->bias[oc_i];
            }

            //printf("Layer %d output (after bias), first 10 elements:\n", oc_i);
            //for (int idx = 0; idx < 10; idx++) {
            //    printf("%f ", output[b * oc * input_length + oc_i * input_length + idx]);
            //}
            //printf("\n");

            // Ora applichiamo la convoluzione causale per ogni canale in input.
            // La convoluzione scorrerà nel tempo e, per ogni istante t,
            // considererà i valori precedenti a t secondo il kernel e la dilatazione.
            for (int t = 0; t < input_length; t++) {
                for (int ic_i = 0; ic_i < ic; ic_i++) {
                    float val = 0.0f; // Accumulator per la somma dei contributi del kernel
                    // Iteriamo sul kernel (di lunghezza ks)
                    for (int k = 0; k < ks; k++) {
                        // Calcoliamo l'indice temporale sull'input corrispondente all'offset k
                        // tenendo conto della dilazione (dilation).
                        int t_in = t - ((ks - 1 - k) * dilation);
                        // Causale: se t_in < 0 significa che stiamo cercando di usare informazioni future,
                        // che non vanno considerate. Quindi aggiungiamo solo se t_in >= 0.
                        if (t_in >= 0) {
                            // Estraiamo il peso corrispondente al kernel (oc_i, ic_i, k)
                            // L'ordine dei pesi è [out_channels, in_channels, kernel_size],
                            // quindi l'indice: oc_i*(ic*ks) + ic_i*(ks) + k.
                            float w = layer->weight[oc_i * ic * ks + ic_i * ks + k];

                            // Estraiamo il valore di input corrispondente (b, ic_i, t_in)
                            // L'ordine dell'input è [batch, in_channels, input_length],
                            // quindi l'indice: b*(ic*input_length) + ic_i*(input_length) + t_in.
                            float x = input[b * ic * input_length + ic_i * input_length + t_in];

                            // Accumuliamo la moltiplicazione w*x
                            val += w * x;
                        }
                    }
                    // Aggiungiamo il contributo del kernel per questo canale di input all'output;
                    output[b * oc * input_length + oc_i * input_length + t] += val;
                }
            }
        }
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    if (INFO_MODE) printf("     Tempo di esecuzione per un batch: %.6f secondi\n\n", elapsed_time);

    // Alla fine di questo processo, 'output' contiene la somma dei bias e delle convoluzioni
    // per tutti i canali in input, producendo così i canali di output.
}
