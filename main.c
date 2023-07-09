#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "mat.h"
#include "nn.h"
#include "sample.h"
#include "helper.h"

int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    int sizes[3] = {28*28, 100, 10};
    NN* nn = NN_malloc(sizes, 3);
    // nn_initialize_standard_norm(nn);
    nn_initialize_fanin(nn);
    nn_set_loss(nn, NN_CE_LOSS);
    nn_set_output_layer_type(nn, NN_SOFTMAX_OUT);

    float lr = 0.5; // learning rate
    float lambda = 0.00; // L2 regularization
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