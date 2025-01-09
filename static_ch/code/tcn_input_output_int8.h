#ifndef TCN_IO_REF_INT8_H
#define TCN_IO_REF_INT8_H

#include <stdint.h>

static const int8_t INPUT_REF[32] = {
    -8, 8, -8, -11, 3, -7, 5, 11, 
    -17, 14, -17, 33, 7, 24, -11, 27, 
    -2, 12, 11, -4, 18, 12, 29, 2, 
    3, 23, -9, 9, -1, -44, -2, -16
};

static const int8_t OUTPUT_REF[1] = {
    10
};

#endif // TCN_IO_REF_INT8_H
