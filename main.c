#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "mat.h"
#include "nn.h"
#include "sample.h"
#include "helper.h"
#include "conv.h"

int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    // int conv_side = 5;
    // int pool_size = 2;
    // int stride = 1;
    // int feature_count = 20;
    // int in_side = 28;

    // float lr = 0.1; // learning rate
    // float lambda = 0.0; // L2 regularization
    // int epochs = 60;
    // int minibatch_size = 10;
    // int training_samples_count = 6000;
    // int test_samples_count = 1000;

    // MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count);
    // MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count);

    // Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count);
    // Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count);
    
    // int sizes[3] = {12*12*feature_count, 100, 10};
    // ConvNN* convnn = convnn_malloc(sizes, 3, conv_side, pool_size, stride, feature_count, in_side);
    // nn_initialize_fanin(convnn->nn);
    // convpool_initialize_standard_norm(convnn->convpool);
    // nn_set_loss(convnn->nn, NN_CE_LOSS);
    // nn_set_output_layer_type(convnn->nn, NN_SOFTMAX_OUT);

    // convnn_sgd(convnn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    int sizes[3] = {28*28, 100, 10};
    NN* nn = nn_malloc(sizes, 3);
    // nn_initialize_standard_norm(nn);
    nn_initialize_fanin(nn);
    nn_set_loss(nn, NN_CE_LOSS);
    nn_set_output_layer_type(nn, NN_SIGMOID_OUT);

    float lr = 0.5; // learning rate
    float lambda = 0.0; // L2 regularization
    int epochs = 60;
    int minibatch_size = 10;
    int training_samples_count = 60000;
    int test_samples_count = 10000;

    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count);

    Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count);
    Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count);

    nn_sgd(nn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    return 0;
}