#ifndef TCN_IO_REF_INT16_H
#define TCN_IO_REF_INT16_H

#include <stdint.h>

static const int16_t INPUT_REF[32] = {
    183, -195, -237, 96, -304, -194, 208, 362, 
    241, 87, -282, -393, 29, 116, -52, -375, 
    69, -185, -267, -214, 209, -110, 180, 34, 
    -106, 181, 52, 266, -61, 170, 190, 323, 

};

static const int16_t OUTPUT_REF[1] = {
    [1724], 
};

#endif // TCN_IO_REF_INT16_H
