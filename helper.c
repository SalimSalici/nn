#include "helper.h"
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

float standard_gauss() {
    float x;
    do {
        x = (float)rand() / RAND_MAX;
    } while (x == 0.0);
    float y = (float)rand() / RAND_MAX;
    float z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
    return z;
}

int random_range_int(int min, int max) {
    assert(min < max);
    int range = max - min;
    float rf = (float)rand() / RAND_MAX;

    return min + round(rf * (float)range);
}

// Takes in and array of pointers and knuth-shuffles the array
void shuffle_pointers(void* array[], int count) {
    for (size_t i = 0; i < count - 2; i++) {
        size_t j = random_range_int(i, count - 1);
        assert(j >= i && j < count);
        
        void* tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

int max (int a, int b) {
    return a > b ? a : b;
}

int min (int a, int b) {
    return a < b ? a : b;
}

int is_divisible(int x, int y) {
    return fabs(fmod((float)x, (float)y)) < 0.000001;
}