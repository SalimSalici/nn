#include "helper.h"
#include <math.h>
#include <stdlib.h>

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

float gauss() {
    float x;
    do {
        x = (float)rand() / RAND_MAX;
    } while (x == 0.0);
    float y = (float)rand() / RAND_MAX;
    float z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
    return z;
}

int max (int a, int b) {
    return a > b ? a : b;
}

int min (int a, int b) {
    return a < b ? a : b;
}