#ifndef _CONV_H
#define _CONV_H

#include "mat.h"
#include "nn.h"

typedef struct ConvPool {
    int stride;
    int conv_side;
    int pool_side;
    int feature_count;
    int in_side;
    Mat** weights;
    float* biases;
    int featuremap_side;
    int maxpoolmap_side;
} ConvPool;

typedef struct ConvNN {
    NN* nn;
    ConvPool* convpool;
} ConvNN;

ConvPool* convpool_malloc(int conv_side, int pool_side, int stride, int feature_count, int in_side);
void convpool_free(ConvPool* convpool);
ConvNN* convnn_malloc(int nn_sizes[], int nn_num_layers, int conv_side, int pool_side, int stride, int feature_count, int in_side);
void convnn_free(ConvNN* convnn);
ConvPool* convpool_initialize_standard_norm(ConvPool* convpool);
ConvNN* convnn_sgd(ConvNN* convnn, Sample** training_samples, int training_count, int epochs, int minibatch_size, float lr, float lambda, Sample** test_samples, int test_count);
Mat* convnn_feedforward(ConvNN* convnn, Mat* inputs);

#endif