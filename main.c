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

int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    goto_set_num_threads(5);
    openblas_set_num_threads(5);

    NN* nn = nn_malloc(NN_NLL_LOSS);
    nn_add_layer(nn, layer_malloc(0, 28*28, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(28*28, 200, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(200, 100, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(100, 10, NN_SOFTMAX_ACT, 0.0));
    nn_initialize_xavier(nn);

    float lr = 0.01; // learning rate
    float lambda = 0.0; // L2 regularization
    int epochs = 60;
    int minibatch_size = 2;
    int training_samples_count = 60000;
    int test_samples_count = 10000;
    float black = 0;
    float white = 1;

    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count, black, white);
    Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count, black, white);

    nn_sgd(nn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    return 0;
}