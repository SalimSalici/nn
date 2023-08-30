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

Cifar10Sample* cifar10_load_samples_CHW(char* data_file_name, size_t offset, size_t count, float black, float white) {

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

Cifar10Sample* cifar10_load_samples_HWC(char* data_file_name, size_t offset, size_t count, float black, float white) {

    uint8_t buffer[32*32*3];

    FILE* datafileptr = fopen(data_file_name, "rb");

    size_t row_size = 32*32*3 + 1;
    fseek(datafileptr, offset * row_size, SEEK_SET);

    Cifar10Sample* samples = (Cifar10Sample*)malloc(sizeof(Cifar10Sample) * count);

    for (size_t i = 0; i < count; i++) {
        fread(&(samples[i].label), 1, 1, datafileptr);
        fread(&buffer, sizeof(uint8_t)*32*32*3, 1, datafileptr);
        for (size_t j = 0; j < 32*32; j++) {
            samples[i].data[3*j] = (float)buffer[j] / 255.0;
            samples[i].data[3*j+1] = (float)buffer[j + 32*32] / 255.0;
            samples[i].data[3*j+2] = (float)buffer[j + 32*32*2] / 255.0;

            samples[i].data[3*j] = black + ((white - black) * samples[i].data[3*j]);
            samples[i].data[3*j+1] = black + ((white - black) * samples[i].data[3*j+1]);
            samples[i].data[3*j+2] = black + ((white - black) * samples[i].data[3*j+2]);
        }
        // for (size_t j = 0; j < 32*32*3; j++) {
        //     samples[i].data[j] = (float)buffer[j] / 255.0;
        //     samples[i].data[j] = black + ((white - black) * samples[i].data[j]);
        // }
    }

    fclose(datafileptr);

    return samples;
}

Sample** cifar10_samples_to_samples(Cifar10Sample* cifar10_samples, int count, float black, float white) {
    Sample** samples = (Sample**)malloc(sizeof(Sample*) * count);

    for (int i = 0; i < count; i++) {
        samples[i] = (Sample*)malloc(sizeof(Sample));
        
        samples[i]->inputs = mat_malloc_nodata(32*32*3, 1);
        samples[i]->inputs->data = cifar10_samples[i].data;

        samples[i]->outputs = mat_malloc(10, 1);
        mat_fill(samples[i]->outputs, black);
        samples[i]->outputs->data[cifar10_samples[i].label] = white;
    }
    return samples;
}

Sample** cifar10_samples_to_samples_start_from(Sample** samples, Cifar10Sample* cifar10_samples, int start_from, int count, float black, float white) {
    for (int i = 0; i < count; i++) {

        int samples_i = start_from + i;

        samples[samples_i] = (Sample*)malloc(sizeof(Sample));
        
        samples[samples_i]->inputs = mat_malloc_nodata(32*32*3, 1);
        samples[samples_i]->inputs->data = cifar10_samples[i].data;

        samples[samples_i]->outputs = mat_malloc(10, 1);
        mat_fill(samples[samples_i]->outputs, black);
        samples[samples_i]->outputs->data[cifar10_samples[i].label] = white;
    }
    return samples;
}

void cifar10_print_image_CHW(float* image) {
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

void cifar10_print_image_HWC(float* image) {
    for (int i = 0; i < 32*32*3; i += 3) {
        float greyscale = image[i] + image[i+1] + image[i+2];
        greyscale /= 3;
        int shade_idx = (int)round(greyscale * (float)(CIFAR_SHADES_LEN - 1));
        char c = __cifar10_loader_shades[shade_idx];
        printf("%c", c);
        if (i/3 % 32 == 31)
            printf("\n");
    }
}

// Sample** mnist_samples_to_samples(Cifar10Sample* mnist_samples, int count, float black, float white);
// void mnist_print_image(float* image);

#endif