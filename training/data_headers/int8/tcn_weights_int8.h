#ifndef TCN_WEIGHTS_INT8_H
#define TCN_WEIGHTS_INT8_H

#include <stdint.h>

static const int8_t layer0_weights[] = {
    6, -4, 0, 0, 0, 0, -20, -7, 
    -40, 0, 11, -11, 16, -9, -4, 0, 
    0, 0, -21, -28, -38, 0, -10, 9
};

static const int8_t layer0_biases[] = {
    -15, 52
};

static const int8_t layer1_weights[] = {
    0, -1, -3, 0, -1, -2, -2, -5, 
    -4, -2, -4, -4
};

static const int8_t layer1_biases[] = {
    64, 40
};

static const int8_t layer2_weights[] = {
    1, 3, 27, -3, 9, 23
};

static const int8_t layer2_biases[] = {
    93
};

#endif // TCN_WEIGHTS_INT8_H
