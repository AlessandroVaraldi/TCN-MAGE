#ifndef TCN_WEIGHTS_INT16_H
#define TCN_WEIGHTS_INT16_H

#include <stdint.h>

static const int16_t layer0_weights[] = {
    101, -68, -8, -1, -3, 4, -331, -120, 
    -642, 1, 176, -177, 266, -157, -72, -4, 
    1, -3, -343, -450, -615, 5, -164, 155
};

static const int16_t layer0_biases[] = {
    -247, 840
};

static const int16_t layer1_weights[] = {
    0, -18, -54, 0, -18, -33, -39, -93, 
    -77, -43, -79, -73
};

static const int16_t layer1_biases[] = {
    1031, 653
};

static const int16_t layer2_weights[] = {
    18, 54, 441, -52, 153, 382
};

static const int16_t layer2_biases[] = {
    1492
};

#endif // TCN_WEIGHTS_INT16_H
