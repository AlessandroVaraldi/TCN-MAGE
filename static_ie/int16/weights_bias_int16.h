#ifndef WEIGHTS_BIAS_INT16_H
#define WEIGHTS_BIAS_INT16_H

#include <stdint.h>

static const int16_t WEIGHTS[] = {
0, 0, 1, 0, 0, 0, -1, 1, 
    0, 0, 3, 4, 0, 0, 0, -2, 
    7, 80, -25, -77, -14, 40, -31, 30, 
    -243, 326, -535, -419, 107, 178, -37, 27, 
    128, 69, 109, 96, 46, -244, 246, 150
};

static const int16_t BIASES[] = {
129, 151, 86, -59, -202, -164, -41, 235
};

#endif // WEIGHTS_BIAS_INT16_H
