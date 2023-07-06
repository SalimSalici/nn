#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "mat.h"
#include "nn.h"
 
int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    int sizes[4] = {28*28, 100, 30, 10};
    NN* nn = NN_malloc(sizes, 4);
    nn_initialize_standard_norm(nn);

    float lr = 1.0;
    int epochs = 30;
    int minibatch_size = 10;
    int training_samples_count = 60000;
    int test_samples_count = 10000;

    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count);

    nn_sgd(nn, training_data, training_samples_count, epochs, minibatch_size, lr, test_data, test_samples_count);

    return 0;
}