#ifndef TCN_WEIGHTS_INT16_H
#define TCN_WEIGHTS_INT16_H

#include <stdint.h>

static const int16_t layer0_weights[] = {
    -16, -43, 38, -103, 57, 61, 338, 398, 
    499, -4, -2, -4, -61, -16, -34, 6, 
    -36, -25, 426, 298, 517, 14, 2, 17, 

};

static const int16_t layer0_biases[] = {
    732, 1009, 
};

static const int16_t layer1_weights[] = {
    -46, -75, -12, -27, -1, -116, 60, 91, 
    124, 24, -34, 94, 
};

static const int16_t layer1_biases[] = {
    0, 1071, 
};

static const int16_t layer2_weights[] = {
    9, 108, -32, -53, 45, 87, 
};

static const int16_t layer2_biases[] = {
    794, 
};

#endif // TCN_WEIGHTS_INT16_H
