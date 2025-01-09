#ifndef TCN_IO_REF_INT8_H
#define TCN_IO_REF_INT8_H

#include <stdint.h>

static const int8_t INPUT_REF[32] = {
    11, -12, -14, 6, -19, -12, 13, 22, 
    15, 5, -17, -24, 1, 7, -3, -23, 
    4, -11, -16, -13, 13, -6, 11, 2, 
    -6, 11, 3, 16, -3, 10, 11, 20, 

};

static const int8_t OUTPUT_REF[1] = {
    [107], 
};

#endif // TCN_IO_REF_INT8_H
