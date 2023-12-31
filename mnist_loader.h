#ifndef _MNIST_LOADER_H
#define _MNIST_LOADER_H

#include <stdint.h>
#include "sample.h"

typedef struct MnistSample {
    float data[28*28];
    uint8_t label;
} MnistSample;

uint8_t* mnist_load_train_images_raw(char* file_name);
float* mnist_load_train_images(char* file_name);
MnistSample* mnist_load_samples(char* data_file_name, char* labels_file_name, size_t offset, size_t count, float black, float white);
Sample** mnist_samples_to_samples(MnistSample* mnist_samples, int count, float black, float white);
Sample** mnist_samples_to_autoenc_samples(MnistSample* mnist_samples, int count, float black, float white);
Sample** mnist_samples_to_conv_samples(MnistSample* mnist_samples, int count, float black, float white);
void mnist_print_image(float* image);

#endif