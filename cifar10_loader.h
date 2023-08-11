#ifndef _CIFAR10_LOADER_H
#define _CIFAR10_LOADER_H

#include <stdint.h>
#include "sample.h"
#include "tensor.h"

#define CIFAR_SHADES_LEN 10

// const char __cifar10_loader_shades[CIFAR_SHADES_LEN] = {' ','.', ':', '-', '=', '+', '*', '#', '%', '@'};
const char __cifar10_loader_shades[CIFAR_SHADES_LEN+1] = " .:-=+*#%@";

typedef struct Cifar10Sample {
    float data[32*32*3];
    uint8_t label;
} Cifar10Sample;

Cifar10Sample* cifar10_load_samples(char* data_file_name, size_t offset, size_t count, float black, float white) {


    uint8_t buffer[32*32*3];

    FILE* datafileptr = fopen(data_file_name, "rb");

    size_t row_size = 32*32*3 + 1;
    fseek(datafileptr, offset * row_size, SEEK_SET);

    Cifar10Sample* samples = (Cifar10Sample*)malloc(sizeof(Cifar10Sample) * count);

    for (size_t i = 0; i < count; i++) {
        fread(&(samples[i].label), 1, 1, datafileptr);
        fread(&buffer, sizeof(uint8_t)*32*32*3, 1, datafileptr);
        for (size_t j = 0; j < 32*32*3; j++) {
            samples[i].data[j] = (float)buffer[j] / 255.0;
            samples[i].data[j] = black + ((white - black) * samples[i].data[j]);
        }
    }

    fclose(datafileptr);

    return samples;
}

void cifar10_print_image(float* image) {
    for (int i = 0; i < 32*32; i++) {
        float greyscale = image[i] + image[32*32 + i] + image[32*32*2 + i];
        greyscale /= 3;
        int shade_idx = (int)round(greyscale * (float)(CIFAR_SHADES_LEN - 1));
        char c = __cifar10_loader_shades[shade_idx];
        printf("%c", c);
        if (i % 32 == 31)
            printf("\n");
    }
}

// Sample** mnist_samples_to_samples(Cifar10Sample* mnist_samples, int count, float black, float white);
// void mnist_print_image(float* image);

#endif