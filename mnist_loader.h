#ifndef _MNIST_LOADER_H
#define _MNIST_LOADER_H

#include <stdint.h>

#define MNIST_SHADES_LEN 5

typedef struct MnistSample {
    float data[28*28];
    uint8_t label;
} MnistSample;

uint32_t map_uint32(uint32_t in);
uint8_t* mnist_load_train_images_raw(char* file_name);
float* mnist_load_train_images(char* file_name);
MnistSample* mnist_load_samples(char* data_file_name, char* labels_file_name, size_t offset, size_t count);
void mnist_print_image(float* image);

#endif