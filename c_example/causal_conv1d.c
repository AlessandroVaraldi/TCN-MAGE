#include "causal_conv1d.h"
#include <string.h> // memset

void causal_conv1d_forward(
    const CausalConv1dLayer *layer,
    const float *input,
    float *output,
    int batch,
    int in_channels,
    int input_length
) {
    // output: [batch, out_channels, time]
    // inizializzo output a zero
    memset(output, 0, sizeof(float)*batch*layer->out_channels*input_length);

    int dilation = layer->dilation;
    int ks = layer->kernel_size;
    int oc = layer->out_channels;
    int ic = layer->in_channels; // dovrebbe coincidere con in_channels passato

    for (int b = 0; b < batch; b++) {
        for (int oc_i = 0; oc_i < oc; oc_i++) {
            // Aggiungo bias
            for (int t = 0; t < input_length; t++) {
                output[b*oc*input_length + oc_i*input_length + t] = layer->bias[oc_i];
            }

            // Convoluzione causale
            for (int ic_i = 0; ic_i < ic; ic_i++) {
                for (int t = 0; t < input_length; t++) {
                    float val = 0.0f;
                    for (int k = 0; k < ks; k++) {
                        int t_in = t - k*dilation;
                        if (t_in >= 0) {
                            float w = layer->weight[oc_i*ic*ks + ic_i*ks + k];
                            float x = input[b*ic*input_length + ic_i*input_length + t_in];
                            val += w * x;
                        }
                    }
                    output[b*oc*input_length + oc_i*input_length + t] += val;
                }
            }
        }
    }
}
