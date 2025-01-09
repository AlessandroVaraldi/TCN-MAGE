#ifndef TCN_WEIGHTS_INT16_H
#define TCN_WEIGHTS_INT16_H

#include <stdint.h>

static const int16_t layer0_weights[] = {
    -60, 7, 41, -3, 2, 1, -52, -260, 
    -337, 45, 78, -124, 243, -119, -73, -2, 
    -1, 2, -370, -370, -319, -24, -43, 65
};

static const int16_t layer0_biases[] = {
    -176, 1578
};

static const int16_t layer1_weights[] = {
    1, -24, -83, 0, -4, -47, 38, -62, 
    -236, -11, -5, -130
};

static const int16_t layer1_biases[] = {
    909, 326
};

static const int16_t layer2_weights[] = {
    18, 274, 667, -52, 355, -20
};

static const int16_t layer2_biases[] = {
    1434
};

#endif // TCN_WEIGHTS_INT16_H
