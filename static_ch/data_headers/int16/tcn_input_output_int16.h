#ifndef TCN_IO_REF_INT16_H
#define TCN_IO_REF_INT16_H

#include <stdint.h>

static const int16_t INPUT_REF[32] = {
    -141, 136, -135, -182, 48, -119, 82, 185, 
    -274, 228, -278, 529, 121, 390, -188, 432, 
    -32, 197, 191, -72, 294, 195, 470, 41, 
    48, 379, -147, 151, -30, -714, -32, -258, 

};

static const int16_t OUTPUT_REF[1] = {
    [4259], 
};

#endif // TCN_IO_REF_INT16_H
