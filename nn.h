#ifndef _NN_H
#define _NN_H

#include "mat.h"
#include "sample.h"
#include "mnist_loader.h"

typedef struct NN {
    int num_layers;
    int* sizes;
    Mat** biases;
    Mat** weights;
} NN;

float standard_norm_initializer_cb(float cur, int row, int col);
float zero_initializer_cb(float cur, int row, int col);
NN* NN_malloc(int sizes[], int num_layers);
void NN_free(NN* nn);
NN* nn_initialize_standard_norm(NN* nn);
NN* backprop(NN* nn, Mat* inputs, Mat* outputs);
void nn_print_biases(NN* nn);
void nn_print_weights(NN* nn);
Mat* nn_feedforward(NN* nn, Mat* inputs);
NN* nn_sgd(NN* nn, Sample** training_samples, int training_count, int epochs, int minibatch_size, float lr, Sample** test_samples, int test_count);
float nn_mat_sigmoid_cb(float cur, int row, int col);
float sigmoid(float z);

#endif