// tcn_layer.h
#ifndef TCN_LAYER_H
#define TCN_LAYER_H

void relu(float* input, float* output, int len);
void tcn_layer(float* input, float* output, float* kernel, int input_len, int kernel_size, int stride, int padding);

#endif // TCN_LAYER_H
