#ifndef TCN_NETWORK_PARAMS_H
#define TCN_NETWORK_PARAMS_H

#include <stdint.h>

#define NUM_LAYERS 5
#define INPUT_DIM 4
#define TIME_LENGTH 300
#define FIXED_POINT 12
#define MAX_HIDDEN_DIM 256

#define NUM_CLASSES 4

static const int32_t HIDDEN_DIMS[] = { 32, 64, 128, 256, 4 };
static const int32_t KERNEL_SIZES[] = { 3, 3, 3, 1, 1 };
static const int32_t DILATIONS[] = { 2, 4, 8, 1, 1 };
static const int32_t RELU_FLAGS[] = { 1, 1, 1, 1, 0 };
static const int32_t MAXPOOL_FLAGS[] = { 1, 1, 1, 0, 0 };
static const int32_t DROPOUT_FLAGS[] = { 0, 0, 1, 1, 0 };
static const int32_t GAP_FLAGS[] = { 0, 0, 0, 0, 1 };
static const int32_t SOFTMAX_FLAGS[] = { 0, 0, 0, 0, 1 };

#endif // TCN_NETWORK_PARAMS_H
