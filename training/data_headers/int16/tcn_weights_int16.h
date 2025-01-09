#ifndef TCN_WEIGHTS_INT16_H
#define TCN_WEIGHTS_INT16_H

#include <stdint.h>

static const int16_t layer0_weights[] = {
    -74, 54, 45, -11, -20, -65, 20, -55, 
    -10, -68, 0, 8, 145, -6, 39, -3, 
    0, -1, -509, -458, -563, 12, -31, 15
};

static const int16_t layer0_biases[] = {
    5, 136
};

static const int16_t layer1_weights[] = {
    -2, 9, 5, -8, -3, 0, 65, 71, 
    -33, -132, 33, 79
};

static const int16_t layer1_biases[] = {
    801, 1
};

static const int16_t layer2_weights[] = {
    18, 221, 698, -52, -67, -161
};

static const int16_t layer2_biases[] = {
    464
};

#endif // TCN_WEIGHTS_INT16_H
