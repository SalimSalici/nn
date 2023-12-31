#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "mat.h"
#include "nn.h"
#include "sample.h"
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"

#include "tensor.h"
#include "mec.h"
#include "conv2d.h"
#include "maxpool.h"
#include "cmpl.h"

#include "cnn.h"

int main(int argc, char* argv[]) {

    srand(time(NULL) * time(NULL));

    goto_set_num_threads(2);
    openblas_set_num_threads(2);

    CNN* cnn = cnn_malloc();

    // NN* nn = nn_malloc(NN_NLL_LOSS);
    // nn_add_layer(nn, layer_malloc(0, 20*12*12, NN_NONE_ACT, 0.0));
    // nn_add_layer(nn, layer_malloc(20*12*12, 100, NN_RELU_ACT, 0.0));
    // nn_add_layer(nn, layer_malloc(100, 10, NN_SOFTMAX_ACT, 0.0));
    // nn_initialize_xavier(nn);
    // cnn_add_cmpl_layer(cnn, cmpl_malloc(
    //     28, 28, 1,
    //     5, 5, 1, 20,
    //     2, 2, 2
    // ));
    int featuremaps0 = 20;
    int featuremaps_last = 40;
    NN* nn = nn_malloc(NN_NLL_LOSS);
    nn_add_layer(nn, layer_malloc(0, featuremaps_last*4*4, NN_NONE_ACT, 0.2));
    nn_add_layer(nn, layer_malloc(featuremaps_last*4*4, 500, NN_RELU_ACT, 0.5));
    nn_add_layer(nn, layer_malloc(500, 500, NN_RELU_ACT, 0.5));
    nn_add_layer(nn, layer_malloc(500, 10, NN_SOFTMAX_ACT, 0.0));
    nn_initialize_xavier(nn);
    cnn_add_cmpl_layer(cnn, cmpl_malloc(
        28, 28, 1,
        5, 5, 1, featuremaps0,
        2, 2, 2
    ));
    cnn_add_cmpl_layer(cnn, cmpl_malloc(
        12, 12, featuremaps0,
        5, 5, 1, featuremaps_last,
        2, 2, 2
    ));

    cnn_set_nn(cnn, nn);

    float lr = 0.05; // learning rate
    float lambda = 0.00; // L2 regularization
    int epochs = 60;
    int minibatch_size = 50;
    int training_samples_count = 2400;
    int test_samples_count = 400;
    float black = 0;
    float white = 1;

    // MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    // MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);
    MnistSample* training_data = mnist_load_samples("data/emnist/emnist-digits-train-images-idx3-ubyte", "data/emnist/emnist-digits-train-labels-idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/emnist/emnist-digits-test-images-idx3-ubyte", "data/emnist/emnist-digits-test-labels-idx1-ubyte", 0, test_samples_count, black, white);
    // MnistSample* training_data = mnist_load_samples("data/fashion/train-images.idx3-ubyte", "data/fashion/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    // MnistSample* test_data = mnist_load_samples("data/fashion/t10k-images.idx3-ubyte", "data/fashion/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count, black, white);
    Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count, black, white);

    cnn_sgd(cnn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);
    // cnn_sgd(cnn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, training_samples, training_samples_count);

    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////

    // NN* nn = nn_malloc(NN_NLL_LOSS);
    // nn_add_layer(nn, layer_malloc(0, 28*28, NN_RELU_ACT, 0.2));
    // nn_add_layer(nn, layer_malloc(28*28, 1000, NN_RELU_ACT, 0.5));
    // nn_add_layer(nn, layer_malloc(1000, 10, NN_SOFTMAX_ACT, 0.0));
    // nn_initialize_xavier(nn);

    // float lr = 0.1; // learning rate
    // float lambda = 0.0; // L2 regularization
    // int epochs = 30;
    // int minibatch_size = 50;
    // int training_samples_count = 60000;
    // int test_samples_count = 10000;
    // float black = 0;
    // float white = 1;

    // MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    // MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);
    // // MnistSample* training_data = mnist_load_samples("data/fashion/train-images.idx3-ubyte", "data/fashion/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    // // MnistSample* test_data = mnist_load_samples("data/fashion/t10k-images.idx3-ubyte", "data/fashion/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    // Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count, black, white);
    // Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count, black, white);

    // nn_sgd(nn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    return 0;
}