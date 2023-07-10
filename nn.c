#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "mat.h"
#include "helper.h"
#include "mnist_loader.h"
#include "sample.h"

NN* nn_malloc(int sizes[], int num_layers) {
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

    nn->output_layer_type = NN_SIGMOID_OUT;
    nn->loss_function = NN_MSE_LOSS;

    return nn;
}

void nn_free(NN* nn) {
    free(nn->sizes);
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_free(nn->biases[i]);
        mat_free(nn->weights[i]);
    }
    free(nn->biases);
    free(nn->weights);
    free(nn);
}

NN* nn_set_output_layer_type(NN* nn, int type) {
    nn->output_layer_type = type;
    return nn;
}

NN* nn_set_loss(NN* nn, int loss) {
    nn->loss_function = loss;
    return nn;
}

// Initializes weights and biases sampling from standard normal distribution
NN* nn_initialize_standard_norm(NN* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_fill_func(nn->biases[i], nn->biases[i], mat_standard_norm_filler_cb, NULL);
        mat_fill_func(nn->weights[i], nn->weights[i], mat_standard_norm_filler_cb, NULL);
    }
    return nn;
}

// Initializes the weights of the network based on the fan-in of each layer and biases to 0
NN* nn_initialize_fanin(NN* nn) {
    float* norm_args[2];
    for (int i = 0; i < nn->num_layers - 1; i++) {
        float mean = 0.0;
        float sd = 1 / sqrt(nn->sizes[i]);
        norm_args[0] = &mean;
        norm_args[1] = &sd;
        mat_fill_func(nn->biases[i], nn->biases[i], mat_zero_filler_cb, NULL);
        mat_fill_func(nn->weights[i], nn->weights[i], mat_norm_filler_cb, norm_args);
    }
    return nn;
}

NN* nn_initialize_zero(NN* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_fill_func(nn->biases[i], nn->biases[i], mat_zero_filler_cb, NULL);
        mat_fill_func(nn->weights[i], nn->weights[i], mat_zero_filler_cb, NULL);
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

float nn_mat_sigmoid_cb(float cur, int row, int col, void* func_args) {
    return SIGMOID(cur);
}

float nn_mat_sigmoid_prime_cb(float cur, int row, int col, void* func_args) {
    return SIGMOID_PRIME(cur);
}

float nn_mat_relu_cb(float cur, int row, int col, void* func_args) {
    return cur > 0.0 ? cur : 0.0;
}

float nn_mat_relu_prime_cb(float cur, int row, int col, void* func_args) {
    return cur > 0.0 ? 1 : 0.0;
}

Mat* nn_softmax(Mat* z) {
    assert(z->cols == 1);
    Mat* res = mat_malloc(z->rows, 1);

    float z_max = mat_max(z);
    float den = exp(z->data[0] - z_max);

    for (int i = 1; i < z->rows; i++)
        den += exp(z->data[i] - z_max);
    for (int i = 0; i < z->rows; i++)
        res->data[i] = exp(z->data[i] - z_max) / den;

    return res;
}

Mat* nn_feedforward(NN* nn, Mat* inputs) {
    assert(inputs->rows == nn->sizes[0]);
    assert(inputs->cols == 1);

    Mat* a = mat_cpy(inputs);

    for (int i = 0; i < nn->num_layers - 2; i++) {
        Mat* z = mat_mult(NULL, nn->weights[i], a);
        z = mat_add(z, z, nn->biases[i]);
        z = mat_fill_func(z, z, nn_mat_sigmoid_cb, NULL);
        mat_free(a);
        a = z;
    }

    size_t last_layer_idx = nn->num_layers - 2;

    Mat* z = mat_mult(NULL, nn->weights[last_layer_idx], a);
    z = mat_add(z, z, nn->biases[last_layer_idx]);
    mat_free(a);

    if (nn->output_layer_type == NN_SIGMOID_OUT) {
        z = mat_fill_func(z, z, nn_mat_sigmoid_cb, NULL);
        a = z;
    } else if (nn->output_layer_type == NN_SOFTMAX_OUT) {
        a = nn_softmax(z);
        free(z);
    }

    return a;
}

NN* nn_backprop(NN* nn, Mat* inputs, Mat* outputs) {

    Mat** zs = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer values before activation function
    Mat** as = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer activations
    Mat** ds = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer errors (delta)

    // Feedforward
    zs[0] = mat_mult(NULL, nn->weights[0], inputs);
    zs[0] = mat_add(zs[0], zs[0], nn->biases[0]);
    as[0] = mat_cpy(zs[0]);
    as[0] = mat_fill_func(as[0], as[0], nn_mat_sigmoid_cb, NULL);

    for (int i = 1; i < nn->num_layers - 1; i++) {
        zs[i] = mat_mult(NULL, nn->weights[i], as[i-1]);
        zs[i] = mat_add(zs[i], zs[i], nn->biases[i]);

        if (i == nn->num_layers - 2) {
            if (nn->output_layer_type == NN_SOFTMAX_OUT) {
                // Apply softmax to last layer
                as[i] = nn_softmax(zs[i]);
            } else if (nn->output_layer_type == NN_SIGMOID_OUT) {
                // Apply sigmoid to last layers
                as[i] = mat_cpy(zs[i]);
                as[i] = mat_fill_func(as[i], as[i], nn_mat_sigmoid_cb, NULL);
            }
        } else {
            // Apply sigmoid to non-last layers
            as[i] = mat_cpy(zs[i]);
            as[i] = mat_fill_func(as[i], as[i], nn_mat_sigmoid_cb, NULL);
        }
    }

    // Backprop
    int last_layer_idx = nn->num_layers - 2;
    ds[last_layer_idx] = mat_sub(NULL, as[last_layer_idx], outputs);

    if (nn->loss_function == NN_MSE_LOSS) {
        zs[last_layer_idx] = mat_fill_func(zs[last_layer_idx], zs[last_layer_idx], nn_mat_sigmoid_prime_cb, NULL);
        ds[last_layer_idx] = mat_hadamard_prod(ds[last_layer_idx], ds[last_layer_idx], zs[last_layer_idx]);
    }

    for (int i = nn->num_layers - 3; i >= 0; i--) {
        Mat* w_next = mat_transpose(NULL, nn->weights[i+1]);

        ds[i] = mat_mult(NULL, w_next, ds[i+1]);
        zs[i] = mat_fill_func(zs[i], zs[i], nn_mat_sigmoid_prime_cb, NULL);
        ds[i] = mat_hadamard_prod(ds[i], ds[i], zs[i]);

        mat_free(w_next);
    }

    NN* g = nn_malloc(nn->sizes, nn->num_layers);

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

NN* nn_update_minibatch(NN* nn, float lr, float lambda, Sample** minibatch, int minibatch_size, int training_count) {

    NN* g = nn_malloc(nn->sizes, nn->num_layers);
    nn_initialize_zero(g);

    for (int i = 0; i < minibatch_size; i++) {
        Mat* inputs = minibatch[i]->inputs;
        Mat* outputs = minibatch[i]->outputs;

        NN* sample_g = nn_backprop(nn, inputs, outputs);
        // NN* sample_g = nn_backprop_softmaxloss(nn, inputs, outputs);

        for (int j = 0; j < nn->num_layers - 1; j++) {
            g->biases[j] = mat_add(g->biases[j], g->biases[j], sample_g->biases[j]);
            g->weights[j] = mat_add(g->weights[j], g->weights[j], sample_g->weights[j]);
        }

        nn_free(sample_g);
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        g->biases[i] = mat_scale(g->biases[i], g->biases[i], lr / ((float)minibatch_size));
        g->weights[i] = mat_scale(g->weights[i], g->weights[i], lr / ((float)minibatch_size));

        // L2 regularization
        Mat* reg_g = mat_scale(NULL, nn->weights[i], lr * lambda / ((float)training_count));
        g->weights[i] = mat_add(g->weights[i], g->weights[i], reg_g);
        mat_free(reg_g);
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

float nn_evaluate(NN* nn, Sample* test_samples[], int test_count) {

    int correct_predictions = 0;

    for (int i = 0; i < test_count; i++) {
        Mat* inputs = test_samples[i]->inputs;
        Mat* outputs = nn_feedforward(nn, inputs);

        int prediction = nn_argmax(outputs->data, outputs->rows);
        int correct = nn_argmax(test_samples[i]->outputs->data, test_samples[i]->outputs->rows);

        if (prediction == correct)
            correct_predictions++;

        mat_free(outputs);
    }

    return (float)correct_predictions / test_count;
}

NN* nn_sgd(NN* nn, Sample** training_samples, int training_count, int epochs, int minibatch_size,
    float lr, float lambda, Sample** test_samples, int test_count) {

    if (nn->output_layer_type == NN_SOFTMAX_OUT && nn->loss_function == NN_MSE_LOSS) {
        printf("This framework doesn't support Mean Squared Error loss function paired with a softmax output layer. "
        "Please, choose another loss function and/or output layer type.\n");
        exit(0);
    }

    printf("Starting SGD. Initial accuracy: %.2f%%\n", nn_evaluate(nn, test_samples, test_count) * 100);
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        clock_t begin = clock();

        shuffle_pointers((void*)training_samples, training_count);

        for (int batch_offset = 0; batch_offset < training_count; batch_offset += minibatch_size) {
            NN* dg = nn_update_minibatch(nn, lr, lambda, training_samples + batch_offset, minibatch_size, training_count);

            for (int i = 0; i < nn->num_layers - 1; i++) {
                nn->biases[i] = mat_sub(nn->biases[i], nn->biases[i], dg->biases[i]);
                nn->weights[i] = mat_sub(nn->weights[i], nn->weights[i], dg->weights[i]);
            }
            nn_free(dg);
            // printf("Minibatch %d ended...\n", batch_offset / minibatch_size);
        }

        clock_t end = clock();
        float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        
        if (test_samples != NULL) {
            float accuracy = nn_evaluate(nn, test_samples, test_count) * 100;
            printf("Epoch %d completed - Epoch time: %.2fs - Accuracy: %.2f%%\n", epoch, time_spent, accuracy);
        } else
            printf("Epoch %d completed.\n", epoch);

    }
    return nn;
}