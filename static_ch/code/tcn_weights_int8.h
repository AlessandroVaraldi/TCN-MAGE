#ifndef TCN_WEIGHTS_INT8_H
#define TCN_WEIGHTS_INT8_H

#include <stdint.h>

static const int8_t layer0_weights[] = {
    -3, 0, 2, 0, 0, 0, -3, -16, 
    -21, 2, 4, -7, 15, -7, -4, 0, 
    0, 0, -23, -23, -19, -1, -2, 4
};

static const int8_t layer0_biases[] = {
    -11, 98
};

static const int8_t layer1_weights[] = {
    0, -1, -5, 0, 0, -2, 2, -3, 
    -14, 0, 0, -8
};

static const int8_t layer1_biases[] = {
    56, 20
};

static const int8_t layer2_weights[] = {
    1, 17, 41, -3, 22, -1
};

static const int8_t layer2_biases[] = {
    89
};

#endif // TCN_WEIGHTS_INT8_H
