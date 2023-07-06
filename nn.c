#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "mat.h"
#include "helper.h"
#include "mnist_loader.h"

#define SIGMOID(z) (1 / (1 + exp(-z)))
#define SIGMOID_PRIME(z) (SIGMOID(z) * (1 - SIGMOID(z)))

float standard_norm_initializer_cb(float cur, int row, int col) {
    return (float)gauss();
}

float zero_initializer_cb(float cur, int row, int col) {
    return 0.0;
}

NN* NN_malloc(int sizes[], int num_layers) {
    NN* nn = (NN*)malloc(sizeof(NN));
    nn->num_layers = num_layers;
    nn->sizes = (int*)malloc(sizeof(int) * num_layers);
    for (int i = 0; i < num_layers; i++)
        nn->sizes[i] = sizes[i];
    
    nn->biases = (Mat**)malloc(sizeof(Mat*) * (num_layers - 1));
    nn->weights = (Mat**)malloc(sizeof(Mat*) * (num_layers - 1));

    for (int i = 0; i < num_layers - 1; i++) {
        int cur_layer_size = nn->sizes[i+1];
        int prev_layer_size = nn->sizes[i];

        nn->biases[i] = mat_malloc(cur_layer_size, 1);
        nn->weights[i] = mat_malloc(cur_layer_size, prev_layer_size);
    }

    return nn;
}

void NN_free(NN* nn) {
    free(nn->sizes);
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_free(nn->biases[i]);
        mat_free(nn->weights[i]);
    }
    free(nn->biases);
    free(nn->weights);
    free(nn);
}

NN* nn_initialize_standard_norm(NN* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_fill_func(nn->biases[i], nn->biases[i], standard_norm_initializer_cb);
        mat_fill_func(nn->weights[i], nn->weights[i], standard_norm_initializer_cb);
    }
    return nn;
}

NN* nn_initialize_zero(NN* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_fill_func(nn->biases[i], nn->biases[i], zero_initializer_cb);
        mat_fill_func(nn->weights[i], nn->weights[i], zero_initializer_cb);
    }
    return nn;
}

void nn_print_biases(NN* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_print(nn->biases[i]);
        printf("\n");
    }
}

void nn_print_weights(NN* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_print(nn->weights[i]);
        printf("\n");
    }
}

float nn_sigmoid(float z) {
    return 1 / (1 + exp(-z));
}

float nn_mat_sigmoid_cb(float cur, int row, int col) {
    return SIGMOID(cur);
}

float nn_mat_sigmoid_prime_cb(float cur, int row, int col) {
    return SIGMOID_PRIME(cur);
}

Mat* nn_feedforward(NN* nn, Mat* inputs) {
    assert(inputs->rows == nn->sizes[0]);
    assert(inputs->cols == 1);

    Mat* a = mat_cpy(inputs);

    for (int i = 0; i < nn->num_layers - 1; i++) {
        Mat* z = mat_mult(NULL, nn->weights[i], a);
        z = mat_add(z, z, nn->biases[i]);
        z = mat_fill_func(z, z, nn_mat_sigmoid_cb);
        mat_free(a);
        a = z;
    }

    return a;
}

NN* nn_backprop(NN* nn, Mat* inputs, Mat* outputs) {

    Mat** zs = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer values before activation function
    Mat** as = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer activations
    Mat** ds = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer errors (delta)

    zs[0] = mat_mult(NULL, nn->weights[0], inputs);
    zs[0] = mat_add(zs[0], zs[0], nn->biases[0]);
    as[0] = mat_cpy(zs[0]);
    as[0] = mat_fill_func(as[0], as[0], nn_mat_sigmoid_cb);

    for (int i = 1; i < nn->num_layers - 1; i++) {
        zs[i] = mat_mult(NULL, nn->weights[i], as[i-1]);
        zs[i] = mat_add(zs[i], zs[i], nn->biases[i]);
        as[i] = mat_cpy(zs[i]);
        as[i] = mat_fill_func(as[i], as[i], nn_mat_sigmoid_cb);
    }

    int last_layer_idx = nn->num_layers - 2;
    ds[last_layer_idx] = mat_sub(NULL, as[last_layer_idx], outputs);
    zs[last_layer_idx] = mat_fill_func(zs[last_layer_idx], zs[last_layer_idx], nn_mat_sigmoid_prime_cb);
    ds[last_layer_idx] = mat_hadamard_prod(ds[last_layer_idx], ds[last_layer_idx], zs[last_layer_idx]);

    for (int i = nn->num_layers - 3; i >= 0; i--) {
        Mat* w_next = mat_transpose(NULL, nn->weights[i+1]);

        ds[i] = mat_mult(NULL, w_next, ds[i+1]);
        zs[i] = mat_fill_func(zs[i], zs[i], nn_mat_sigmoid_prime_cb);
        ds[i] = mat_hadamard_prod(ds[i], ds[i], zs[i]);

        mat_free(w_next);
    }

    NN* g = NN_malloc(nn->sizes, nn->num_layers);

    mat_free(g->biases[0]);
    mat_free(g->weights[0]);
    g->biases[0] = ds[0];
    Mat* a_prev = mat_transpose(NULL, inputs);
    g->weights[0] = mat_mult(NULL, ds[0], a_prev);
    mat_free(a_prev);

    for (int i = 1; i < nn->num_layers - 1; i++) {
        mat_free(g->biases[i]);
        mat_free(g->weights[i]);
        g->biases[i] = ds[i];
        Mat* a_prev = mat_transpose(NULL, as[i - 1]);
        g->weights[i] = mat_mult(NULL, ds[i], a_prev);
        mat_free(a_prev);
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_free(zs[i]);
        mat_free(as[i]);
    }

    free(zs);
    free(as);
    free(ds);

    return g;
}

NN* nn_update_minibatch(NN* nn, float lr, MnistSample* minibatch, int minibatch_size) {

    NN* g = NN_malloc(nn->sizes, nn->num_layers);
    nn_initialize_zero(g);

    for (int i = 0; i < minibatch_size; i++) {

        Mat* inputs = mat_malloc(28*28, 1);
        memcpy(inputs->data, minibatch[i].data, sizeof(float) * 28 * 28);
        Mat* outputs = mat_malloc(10, 1);

        mat_fill_func(outputs, outputs, zero_initializer_cb);
        outputs->data[minibatch[i].label] = 1.0;

        NN* sample_g = nn_backprop(nn, inputs, outputs);

        for (int j = 0; j < nn->num_layers - 1; j++) {
            g->biases[j] = mat_add(g->biases[j], g->biases[j], sample_g->biases[j]);
            g->weights[j] = mat_add(g->weights[j], g->weights[j], sample_g->weights[j]);
        }

        NN_free(sample_g);
        mat_free(inputs);
        mat_free(outputs);
    }

    for (int j = 0; j < nn->num_layers - 1; j++) {
        g->biases[j] = mat_scale(g->biases[j], g->biases[j], lr / ((float)minibatch_size));
        g->weights[j] = mat_scale(g->weights[j], g->weights[j], lr / ((float)minibatch_size));
    }

    return g;
}

int nn_argmax(float* array, int size) {
    int max_index = 0;
    float max_value = array[0];

    for (int i = 1; i < size; i++) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }

    return max_index;
}

float nn_evaluate(NN* nn, MnistSample* test_data, int test_count) {

    int correct_predictions = 0;

    for (int i = 0; i < test_count; i++) {
        Mat* input = mat_malloc(28*28, 1);
        memcpy(input->data, test_data[i].data, sizeof(float) * 28 * 28);

        Mat* output = nn_feedforward(nn, input);
        int prediction = nn_argmax(output->data, output->rows);
        if (prediction == test_data[i].label)
            correct_predictions++;

        mat_free(input);
    }

    return (float)correct_predictions / test_count;
}

NN* nn_sgd(NN* nn, MnistSample* training_data, int training_count, int epochs, int minibatch_size, float lr, MnistSample* test_data, int test_count) {

    printf("Starting SGD. Initial accuracy: %.2f%%\n", nn_evaluate(nn, test_data, test_count) * 100);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // TODO: shuffle training_data

        for (int batch_offset = 0; batch_offset < training_count; batch_offset += minibatch_size) {
            NN* dg = nn_update_minibatch(nn, lr, training_data + batch_offset, minibatch_size);
            for (int i = 0; i < nn->num_layers - 1; i++) {
                nn->biases[i] = mat_sub(nn->biases[i], nn->biases[i], dg->biases[i]);
                nn->weights[i] = mat_sub(nn->weights[i], nn->weights[i], dg->weights[i]);
            }
            NN_free(dg);
            // printf("Minibatch %d ended...\n", batch_offset / minibatch_size);
        }

        
        if (test_data != NULL) {
            float accuracy = nn_evaluate(nn, test_data, test_count) * 100;
            printf("Epoch %d completed. Accuracy: %.2f%%\n", epoch, accuracy);
        } else
            printf("Epoch %d completed.\n", epoch);

    }
    return nn;
}