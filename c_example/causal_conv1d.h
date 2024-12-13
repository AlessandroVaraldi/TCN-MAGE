#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include <stddef.h>

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int dilation;
    float *weight;  // [out_channels, in_channels, kernel_size]
    float *bias;    // [out_channels]
} CausalConv1dLayer;

/**
 * Esegue una convoluzione causale 1D.
 * input: [batch, in_channels, time]
 * output: [batch, out_channels, time]
 */
void causal_conv1d_forward(
    const CausalConv1dLayer *layer,
    const float *input,
    float *output,
    int batch,
    int in_channels,
    int input_length
);

#endif
