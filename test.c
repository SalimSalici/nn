#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "mnist_loader.h"
#include "mat.h"
#include "sample.h"
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"
#include "conv.h"

int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    goto_set_num_threads(1);
    openblas_set_num_threads(1);

    int minibatch_size = 50;
    int feature_count = 20;
    int kernel_stride = 1;
    int maxpool_stride = 2;
    int input_original_side = 28;
    int kernel_side = 5;
    int maxpool_side = 2;

    CPL* cpl = cpl_malloc(minibatch_size, feature_count, kernel_stride, maxpool_stride, input_original_side, kernel_side, maxpool_side);
    NN* nn = nn_malloc(NN_NLL_LOSS);
    nn_add_layer(nn, layer_malloc(0, 12*12*feature_count, NN_NONE_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(12*12*feature_count, 100, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(100, 100, NN_RELU_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(100, 10, NN_SOFTMAX_ACT, 0.0));
    nn_initialize_xavier(nn);

    CNN* cnn = cnn_malloc(cpl, nn, minibatch_size);

    int training_samples_count = 60000;
    int test_samples_count = 10000;
    int epochs = 60;
    float lr = 0.05;
    float lambda = 0;
    float black = 0;
    float white = 1;

    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-lr") == 0) {
            lr = atof(argv[i+1]);
        }

        if (strcmp(argv[i], "-lambda") == 0) {
            lambda = atof(argv[i+1]);
        }

        if (strcmp(argv[i], "-ep") == 0) {
            epochs = atoi(argv[i+1]);
        }
    }

    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count, black, white);
    Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count, black, white);

    for (int i = 0; i < training_samples_count; i++) {
        training_samples[i]->inputs->rows = 28;
        training_samples[i]->inputs->cols = 28;
        training_samples[i]->inputs->down = 28;
    }

    for (int i = 0; i < test_samples_count; i++) {
        test_samples[i]->inputs->rows = 28;
        test_samples[i]->inputs->cols = 28;
        test_samples[i]->inputs->down = 28;
    }

    cnn_im2col_samples(cnn, training_samples, training_samples_count);
    cnn_im2col_samples(cnn, test_samples, test_samples_count);

    cnn_sgd(cnn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    getchar();

    return 0;
}