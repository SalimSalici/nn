#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "cifar10_loader.h"
#include "mat.h"
#include "sample.h"
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"
#include "cnn_old.h"

const char* classes[] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    goto_set_num_threads(1);
    openblas_set_num_threads(1);

    NN* nn = nn_malloc(NN_NLL_LOSS);
    nn_add_layer(nn, layer_malloc(0, 32*32*3, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(32*32*3, 500, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(500, 100, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(100, 10, NN_SOFTMAX_ACT, 0.0));
    nn_initialize_xavier(nn);

    float lr = 0.01; // learning rate
    float lambda = 0.0; // L2 regularization
    int epochs = 60;
    int minibatch_size = 50;
    int training_samples_count = 10000;
    int test_samples_count = 10000;
    float black = 0;
    float white = 1;

    Cifar10Sample* training_data1 = cifar10_load_samples("data/cifar10/data_batch_1.bin", 0, training_samples_count, black, white);
    Cifar10Sample* training_data2 = cifar10_load_samples("data/cifar10/data_batch_2.bin", 0, training_samples_count, black, white);
    Cifar10Sample* training_data3 = cifar10_load_samples("data/cifar10/data_batch_3.bin", 0, training_samples_count, black, white);
    Cifar10Sample* training_data4 = cifar10_load_samples("data/cifar10/data_batch_4.bin", 0, training_samples_count, black, white);
    Cifar10Sample* training_data5 = cifar10_load_samples("data/cifar10/data_batch_5.bin", 0, training_samples_count, black, white);
    Cifar10Sample* test_data = cifar10_load_samples("data/cifar10/test_batch.bin", 0, test_samples_count, black, white);

    Sample** training_samples = (Sample**)malloc(sizeof(Sample*) * training_samples_count * 5);
    training_samples = cifar10_samples_to_samples_start_from(training_samples, training_data1, 0, training_samples_count, black, white);
    training_samples = cifar10_samples_to_samples_start_from(training_samples, training_data2, 10000, training_samples_count, black, white);
    training_samples = cifar10_samples_to_samples_start_from(training_samples, training_data3, 20000, training_samples_count, black, white);
    training_samples = cifar10_samples_to_samples_start_from(training_samples, training_data4, 30000, training_samples_count, black, white);
    training_samples = cifar10_samples_to_samples_start_from(training_samples, training_data5, 40000, training_samples_count, black, white);

    // Sample** training_samples = cifar10_samples_to_samples(training_data, training_samples_count, black, white);
    Sample** test_samples = cifar10_samples_to_samples(test_data, test_samples_count, black, white);

    nn_sgd(nn, training_samples, 50000, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    // for (int x = 0; x < 30; x++) {
    //     cifar10_print_image(training_data[x].data);
    //     printf("%s\n\n", classes[training_data[x].label]);
    // }
    return 0;
}