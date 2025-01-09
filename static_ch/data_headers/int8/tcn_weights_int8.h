#ifndef TCN_WEIGHTS_INT8_H
#define TCN_WEIGHTS_INT8_H

#include <stdint.h>

static const int8_t layer0_weights[] = {
    -1, -2, 2, -6, 3, 3, 21, 24, 
    31, 0, 0, 0, -3, -1, -2, 0, 
    -2, -1, 26, 18, 32, 0, 0, 1, 

};

static const int8_t layer0_biases[] = {
    45, 63, 
};

static const int8_t layer1_weights[] = {
    -2, -4, 0, -1, 0, -7, 3, 5, 
    7, 1, -2, 5, 
};

static const int8_t layer1_biases[] = {
    0, 66, 
};

static const int8_t layer2_weights[] = {
    0, 6, -2, -3, 2, 5, 
};

static const int8_t layer2_biases[] = {
    49, 
};

#endif // TCN_WEIGHTS_INT8_H
