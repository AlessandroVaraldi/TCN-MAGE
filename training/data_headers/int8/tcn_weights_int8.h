#ifndef TCN_WEIGHTS_INT8_H
#define TCN_WEIGHTS_INT8_H

#include <stdint.h>

static const int8_t layer0_weights[] = {
    -4, 3, 2, 0, -1, -4, 1, -3, 
    0, -4, 0, 0, 9, 0, 2, 0, 
    0, 0, -31, -28, -35, 0, -1, 0
};

static const int8_t layer0_biases[] = {
    0, 8
};

static const int8_t layer1_weights[] = {
    0, 0, 0, 0, 0, 0, 4, 4, 
    -2, -8, 2, 4
};

static const int8_t layer1_biases[] = {
    50, 0
};

static const int8_t layer2_weights[] = {
    1, 13, 43, -3, -4, -10
};

static const int8_t layer2_biases[] = {
    29
};

#endif // TCN_WEIGHTS_INT8_H
