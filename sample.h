#ifndef _SAMPLE_H
#define _SAMPLE_H

#include "mat.h"

typedef struct Sample {
    Mat* inputs;
    Mat* outputs;
} Sample;

#endif