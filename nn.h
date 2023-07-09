#ifndef _NN_H
#define _NN_H

#include "mat.h"
#include "sample.h"
#include "mnist_loader.h"

#define NN_SIGMOID_OUT 0
#define NN_SOFTMAX_OUT 1
#define NN_CE_LOSS 0
#define NN_MSE_LOSS 1

typedef struct NN {
    int num_layers;
    int* sizes;
    Mat** biases;
    Mat** weights;
    int output_layer_type;
    int loss_function;
} NN;

NN* NN_malloc(int sizes[], int num_layers);
void NN_free(NN* nn);
NN* nn_initialize_standard_norm(NN* nn);
NN* nn_initialize_fanin(NN* nn);
NN* nn_set_output_layer_type(NN* nn, int type);
NN* nn_set_loss(NN* nn, int loss);
NN* backprop(NN* nn, Mat* inputs, Mat* outputs);
void nn_print_biases(NN* nn);
void nn_print_weights(NN* nn);
Mat* nn_softmax(Mat* z);
Mat* nn_feedforward(NN* nn, Mat* inputs);
NN* nn_sgd(NN* nn, Sample** training_samples, int training_count, int epochs, int minibatch_size, float lr, float labmda, Sample** test_samples, int test_count);
float nn_mat_sigmoid_cb(float cur, int row, int col, void* func_args);
float sigmoid(float z);

#endif