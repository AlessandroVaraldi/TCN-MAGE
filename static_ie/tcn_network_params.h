#ifndef TCN_NETWORK_PARAMS_H
#define TCN_NETWORK_PARAMS_H

#include <stdint.h>

#define NUM_LAYERS 13
#define INPUT_DIM 4
#define TIME_LENGTH 8
#define FIXED_POINT 12
#define MAX_HIDDEN_DIM 20

#define NUM_CLASSES 4

static const int32_t HIDDEN_DIMS[] = { 20, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 4, 4 };
static const int32_t KERNEL_SIZES[] = { 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8 };
static const int32_t DILATIONS[] = { 1, 2, 4, 4, 4, 4, 6, 6, 6, 6, 8, 8, 8 };
static const int32_t RELU_FLAGS[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const int32_t MAXPOOL_FLAGS[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 };
static const int32_t DROPOUT_FLAGS[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static const int32_t GAP_FLAGS[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
static const int32_t SOFTMAX_FLAGS[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

#endif // TCN_NETWORK_PARAMS_H
