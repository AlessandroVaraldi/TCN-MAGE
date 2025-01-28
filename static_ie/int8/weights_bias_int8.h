#ifndef WEIGHTS_BIAS_INT8_H
#define WEIGHTS_BIAS_INT8_H

#include <stdint.h>

static const int8_t WEIGHTS[] = {
0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    1, 10, -3, -10, -2, 5, -4, 4, 
    -30, 41, -67, -52, 13, 22, -5, 3, 
    16, 9, 14, 12, 6, -30, 31, 19
};

static const int8_t BIASES[] = {
16, 19, 11, -7, -25, -21, -5, 29
};

#endif // WEIGHTS_BIAS_INT8_H
