#include "conv.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include "nn.h"
#include "mat.h"
#include "mnist_loader.h"
#include "helper.h"

ConvPool* convpool_malloc(int conv_side, int pool_side, int stride, int feature_count, int in_side) {
    ConvPool* convpool = (ConvPool*)malloc(sizeof(ConvPool));
    convpool->conv_side = conv_side;
    convpool->pool_side = pool_side;
    convpool->stride = stride;
    convpool->feature_count = feature_count;
    convpool->in_side = in_side;
    convpool->weights = (Mat**)malloc(sizeof(Mat*) * feature_count);
    convpool->biases = (float*)malloc(sizeof(float) * feature_count);
    convpool->featuremap_side = convpool->in_side - convpool->conv_side + 1;
    convpool->maxpoolmap_side = convpool->featuremap_side / convpool->pool_side;

    for (int i = 0; i < feature_count; i++) {
        convpool->weights[i] = mat_malloc(conv_side, conv_side);
        // mat_fill_func(convpool->weights[i], convpool->weights[i], mat_standard_norm_filler_cb, NULL);
        // convpool->biases[i] = 0;
    }
    return convpool;
}

void convpool_free(ConvPool* convpool) {
    for (int i = 0; i < convpool->feature_count; i++)
        mat_free(convpool->weights[i]);
    free(convpool->weights);
    free(convpool->biases);
    free(convpool);
}

ConvPool* convpool_initialize_standard_norm(ConvPool* convpool) {
    for (int i = 0; i < convpool->feature_count; i++) {
        // convpool->weights[i] = mat_malloc(convpool->conv_side, convpool->conv_side);
        mat_fill_func(convpool->weights[i], convpool->weights[i], mat_standard_norm_filler_cb, NULL);
        convpool->biases[i] = 0;
    }
    return convpool;
}

ConvPool* convpool_initialize_zero(ConvPool* convpool) {
    for (int i = 0; i < convpool->feature_count; i++) {
        // convpool->weights[i] = mat_malloc(convpool->conv_side, convpool->conv_side);
        mat_fill_func(convpool->weights[i], convpool->weights[i], mat_zero_filler_cb, NULL);
        convpool->biases[i] = 0;
    }
    return convpool;
}

ConvNN* convnn_malloc(int nn_sizes[], int nn_num_layers, int conv_side, int pool_side, int stride, int feature_count, int in_side) {
    ConvNN* convnn = (ConvNN*)malloc(sizeof(ConvNN));
    convnn->nn = nn_malloc(nn_sizes, nn_num_layers);
    convnn->convpool = convpool_malloc(conv_side, pool_side, stride, feature_count, in_side);
    return convnn;
}

void convnn_free(ConvNN* convnn) {
    nn_free(convnn->nn);
    convpool_free(convnn->convpool);
    free(convnn);
}

Mat* convnn_featuremap(ConvNN* convnn, Mat* weights, float bias, Mat* inputs) {
    int feature_side = convnn->convpool->in_side - convnn->convpool->conv_side + 1;
    Mat* featuremap = mat_malloc(feature_side, feature_side);

    inputs->rows = 28;
    inputs->cols = 28;

    for (int r = 0; r < feature_side; r++) {
        for (int c = 0; c < feature_side; c++) {
            float sum = 0;
            for (int i = 0; i < convnn->convpool->conv_side; i++) {
                for (int j = 0; j < convnn->convpool->conv_side; j++) {
                    sum += weights->data[MAT_ELEM_IDX(weights, i, j)] * inputs->data[MAT_ELEM_IDX(inputs, r + i, c + j)];
                }
            }
            featuremap->data[MAT_ELEM_IDX(featuremap, r, c)] = sum + bias;
        }
    }

    return featuremap;
}

float max_from_pool(Mat* featuremap, int pool_side, int pool_r, int pool_c) {
    float max = -INFINITY;
    for (int r = 0; r < pool_side; r++) {
        for (int c = 0; c < pool_side; c++) {
            float cur = featuremap->data[MAT_ELEM_IDX(featuremap, pool_r*pool_side + r, pool_c*pool_side + c)];
            if (cur > max) max = cur;
        }
    }
    return max;
}

Mat* convnn_maxpoolmap(ConvNN* convnn, Mat* featuremap) {
    assert(featuremap->rows ==  featuremap->cols);
    int maxpoolmap_side = featuremap->rows / convnn->convpool->pool_side;
    Mat* maxpoolmap = mat_malloc(maxpoolmap_side, maxpoolmap_side);

    for (int r = 0; r < maxpoolmap_side; r++) {
        for (int c = 0; c < maxpoolmap_side; c++) {
                maxpoolmap->data[MAT_ELEM_IDX(maxpoolmap, r, c)] = max_from_pool(featuremap, convnn->convpool->pool_side, r, c);
        }
    }

    return maxpoolmap;
}

Mat* convnn_nn_inputs_from_maxpools(ConvPool* convpool, Mat* max_poolsmaps[]) {
    Mat* nn_inputs;

    int featuremap_side = convpool->in_side - convpool->conv_side + 1;
    int maxpoolmap_side = featuremap_side / convpool->pool_side;
    int maxpoolmap_units = maxpoolmap_side * maxpoolmap_side;
    nn_inputs = mat_malloc(maxpoolmap_units * convpool->feature_count, 1);

    for (int i = 0; i < convpool->feature_count; i++) {
        memcpy(nn_inputs->data + i * maxpoolmap_units, max_poolsmaps[i]->data, sizeof(float) * maxpoolmap_units);
    }
    return nn_inputs;
}

Mat* convnn_feedforward(ConvNN* convnn, Mat* inputs) {

    ConvPool* convpool = convnn->convpool;

    Mat* features[convpool->feature_count];
    Mat* max_poolsmaps[convpool->feature_count];
    for (int i = 0; i < convpool->feature_count; i++) {
        features[i] = convnn_featuremap(convnn, convpool->weights[i], convpool->biases[i], inputs);
        features[i] = mat_fill_func(features[i], features[i], nn_mat_sigmoid_cb, NULL);
        max_poolsmaps[i] = convnn_maxpoolmap(convnn, features[i]);
    }

    // int featuremap_side = convpool->in_side - convpool->conv_side + 1;
    // int maxpoolmap_side = featuremap_side / convpool->pool_side;
    // int maxpoolmap_units = maxpoolmap_side * maxpoolmap_side;
    // Mat* nn_inputs = mat_malloc(maxpoolmap_units * convpool->feature_count, 1);

    // for (int i = 0; i < convpool->feature_count; i++) {
    //     memcpy(nn_inputs->data + i * maxpoolmap_units, max_poolsmaps[i]->data, sizeof(float) * maxpoolmap_units);
    // }

    Mat* nn_inputs = convnn_nn_inputs_from_maxpools(convpool, max_poolsmaps);

    Mat* res = nn_feedforward(convnn->nn, nn_inputs);

    for (int i = 0; i < convpool->feature_count; i++) {
        mat_free(features[i]);
        mat_free(max_poolsmaps[i]);
    }
    mat_free(nn_inputs);

    return res;
}

ConvNN* convnn_backprop(ConvNN* convnn, Mat* inputs, Mat* outputs) {


    ConvPool* convpool = convnn->convpool;
    NN* nn = convnn->nn;

    Mat* f_zs[convpool->feature_count];
    Mat* f_as[convpool->feature_count];
    Mat* p_as[convpool->feature_count];

    for (int i = 0; i < convnn->convpool->feature_count; i++) {
        f_zs[i] = convnn_featuremap(convnn, convnn->convpool->weights[i], convnn->convpool->biases[i], inputs);
        f_as[i] = mat_fill_func(NULL, f_zs[i], nn_mat_sigmoid_cb, NULL);
        p_as[i] = convnn_maxpoolmap(convnn, f_as[i]);
    }

    Mat** zs = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer values before activation function
    Mat** as = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer activations
    Mat** ds = (Mat**)malloc(sizeof(Mat*) * (nn->num_layers - 1)); // layer errors (delta)

    Mat* nn_inputs = convnn_nn_inputs_from_maxpools(convpool, p_as);

    // Feedforward
    zs[0] = mat_mult(NULL, nn->weights[0], nn_inputs);
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

    Mat* w_next = mat_transpose(NULL, nn->weights[0]);
    Mat* dps_unified = mat_mult(NULL, w_next, ds[0]);
    mat_free(w_next);

    Mat* dps[convpool->feature_count];

    for (int i = 0; i < convpool->feature_count; i++) {
        int maxpoolmap_units = convpool->maxpoolmap_side * convpool->maxpoolmap_side;
        dps[i] = mat_malloc(convpool->maxpoolmap_side, convpool->maxpoolmap_side);
        memcpy(dps[i]->data, dps_unified->data + i * maxpoolmap_units, maxpoolmap_units * sizeof(float));
    }

    Mat* dfs[convpool->feature_count];

    for (int i = 0; i < convpool->feature_count; i++) {
        dfs[i] = mat_malloc(convpool->featuremap_side, convpool->featuremap_side);
        for (int r = 0; r < convpool->featuremap_side; r++) {
            for (int c = 0; c < convpool->featuremap_side; c++) {
                if (f_as[i]->data[MAT_ELEM_IDX(f_as[i], r, c)] == p_as[i]->data[MAT_ELEM_IDX(p_as[i], r/2, c/2)]) {
                    dfs[i]->data[MAT_ELEM_IDX(dfs[i], r, c)] = SIGMOID_PRIME(dps[i]->data[MAT_ELEM_IDX(dps[i], r/2, c/2)]);
                    // mat_fill_func(dfs[i], dfs[i], nn_mat_sigmoid_prime_cb, NULL);
                } else 
                    dfs[i]->data[MAT_ELEM_IDX(dfs[i], r, c)] = 0;
            }
        }
    }

    ConvPool* conv_g = convpool_malloc(convpool->conv_side, convpool->pool_side, convpool->stride, convpool->feature_count, convpool->in_side);
    // convpool_initialize_zero(conv_g);

    inputs->rows = 28;
    inputs->cols = 28;

    // mat_print(f_as[0]);
    // printf("\n\n");
    // mat_print(p_as[0]);
    // printf("\n\n");
    // mat_print(dfs[0]);
    // for (int f = 0; f < conv_g->feature_count; f++) {
    //     mat_print(f_as[f]);
    //     printf("\n\n");
    // }

    // mat_print(inputs);
    // mnist_print_image(inputs->data);

    for (int f = 0; f < conv_g->feature_count; f++) {
        for (int i = 0; i < conv_g->conv_side; i++) {
            for (int j = 0; j < conv_g->conv_side; j++) {
                float sum_w = 0;
                float sum_b = 0;
                for (int r = 0; r < convpool->featuremap_side; r++) {
                    for (int c = 0; c < convpool->featuremap_side; c++) {
                        // printf("(%d)(%d, %d) ", f, r, c);
                        // printf("(%d, %d)\n", r+i, c+j);
                        sum_w += dfs[f]->data[MAT_ELEM_IDX(dfs[f], r, c)] * inputs->data[MAT_ELEM_IDX(inputs, r+i, c+j)];
                        sum_b += dfs[f]->data[MAT_ELEM_IDX(dfs[f], r, c)];
                    }
                }
                // printf("wow\n");
                conv_g->weights[f]->data[MAT_ELEM_IDX(conv_g->weights[f], i, j)] = sum_w;
                conv_g->biases[f] = sum_b;
            }
        }
    }

    NN* nn_g = nn_malloc(nn->sizes, nn->num_layers);

    mat_free(nn_g->biases[0]);
    mat_free(nn_g->weights[0]);
    nn_g->biases[0] = ds[0];
    Mat* a_prev = mat_transpose(NULL, nn_inputs);
    nn_g->weights[0] = mat_mult(NULL, ds[0], a_prev);
    mat_free(a_prev);

    for (int i = 1; i < nn->num_layers - 1; i++) {
        mat_free(nn_g->biases[i]);
        mat_free(nn_g->weights[i]);
        nn_g->biases[i] = ds[i];
        Mat* a_prev = mat_transpose(NULL, as[i - 1]);
        nn_g->weights[i] = mat_mult(NULL, ds[i], a_prev);
        mat_free(a_prev);
    }

    for (int i = 0; i < nn->num_layers - 1; i++) {
        mat_free(zs[i]);
        mat_free(as[i]);
    }

    for (int i = 0; i < convnn->convpool->feature_count; i++) {
        mat_free(f_zs[i]);
        mat_free(f_as[i]);
        mat_free(p_as[i]);
        mat_free(dps[i]);
        mat_free(dfs[i]);
    }

    mat_free(nn_inputs);
    mat_free(dps_unified);

    free(zs);
    free(as);
    free(ds);

    ConvNN* g = (ConvNN*)malloc(sizeof(ConvNN));

    g->nn = nn_g;
    g->convpool = conv_g;

    return g;
}

ConvNN* convnn_update_minibatch(ConvNN* convnn, float lr, float lambda, Sample** minibatch, int minibatch_size, int training_count) {

    ConvNN* g = convnn_malloc(convnn->nn->sizes, convnn->nn->num_layers, convnn->convpool->conv_side,
        convnn->convpool->pool_side,convnn->convpool->stride,
        convnn->convpool->feature_count, convnn->convpool->in_side);
    nn_initialize_zero(g->nn);
    convpool_initialize_zero(g->convpool);

    for (int i = 0; i < minibatch_size; i++) {
        Mat* inputs = minibatch[i]->inputs;
        Mat* outputs = minibatch[i]->outputs;

        ConvNN* sample_g = convnn_backprop(convnn, inputs, outputs);

        for (int j = 0; j < g->nn->num_layers - 1; j++) {
            g->nn->biases[j] = mat_add(g->nn->biases[j], g->nn->biases[j], sample_g->nn->biases[j]);
            g->nn->weights[j] = mat_add(g->nn->weights[j], g->nn->weights[j], sample_g->nn->weights[j]);
        }

        for (int f = 0; f < g->convpool->feature_count; f++) {
            g->convpool->weights[f] = mat_add(g->convpool->weights[f], g->convpool->weights[f], sample_g->convpool->weights[f]);
            g->convpool->biases[f] += sample_g->convpool->biases[f];
        }

        convnn_free(sample_g);
    }

    for (int i = 0; i < g->nn->num_layers - 1; i++) {
        g->nn->biases[i] = mat_scale(g->nn->biases[i], g->nn->biases[i], lr / ((float)minibatch_size));
        g->nn->weights[i] = mat_scale(g->nn->weights[i], g->nn->weights[i], lr / ((float)minibatch_size));

        // L2 regularization
        Mat* reg_g = mat_scale(NULL, g->nn->weights[i], lr * lambda / ((float)training_count));
        g->nn->weights[i] = mat_add(g->nn->weights[i], g->nn->weights[i], reg_g);
        mat_free(reg_g);
    }

    for (int f = 0; f < g->convpool->feature_count; f++) {
        g->convpool->weights[f] = mat_scale(g->convpool->weights[f], g->convpool->weights[f], lr / ((float)minibatch_size));
        g->convpool->biases[f] /= (float)minibatch_size;
    }

    return g;
}

float convnn_evaluate(ConvNN* convnn, Sample* test_samples[], int test_count) {

    int correct_predictions = 0;

    for (int i = 0; i < test_count; i++) {
        Mat* inputs = test_samples[i]->inputs;
        Mat* outputs = convnn_feedforward(convnn, inputs);

        int prediction = nn_argmax(outputs->data, outputs->rows);
        int correct = nn_argmax(test_samples[i]->outputs->data, test_samples[i]->outputs->rows);

        if (prediction == correct)
            correct_predictions++;

        mat_free(outputs);
    }

    return (float)correct_predictions / test_count;
}

ConvNN* convnn_sgd(ConvNN* convnn, Sample** training_samples, int training_count, int epochs, int minibatch_size,
    float lr, float lambda, Sample** test_samples, int test_count) {

    if (convnn->nn->output_layer_type == NN_SOFTMAX_OUT && convnn->nn->loss_function == NN_MSE_LOSS) {
        printf("This framework doesn't support Mean Squared Error loss function paired with a softmax output layer. "
        "Please, choose another loss function and/or output layer type.\n");
        exit(0);
    }

    printf("Starting SGD. Initial accuracy: %.2f%%\n", convnn_evaluate(convnn, test_samples, test_count) * 100);
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        clock_t begin = clock();

        shuffle_pointers((void*)training_samples, training_count);

        for (int batch_offset = 0; batch_offset < training_count; batch_offset += minibatch_size) {
            ConvNN* dg = convnn_update_minibatch(convnn, lr, lambda, training_samples + batch_offset, minibatch_size, training_count);

            for (int i = 0; i < dg->nn->num_layers - 1; i++) {
                convnn->nn->biases[i] = mat_sub(convnn->nn->biases[i], convnn->nn->biases[i], dg->nn->biases[i]);
                convnn->nn->weights[i] = mat_sub(convnn->nn->weights[i], convnn->nn->weights[i], dg->nn->weights[i]);
            }

            for (int f = 0; f < dg->convpool->feature_count - 1; f++) {
                convnn->convpool->weights[f] = mat_sub(convnn->convpool->weights[f], convnn->convpool->weights[f], dg->convpool->weights[f]);
                convnn->convpool->biases[f] -= dg->convpool->biases[f];
            }

            convnn_free(dg);
            // printf("Minibatch %d ended...\n", batch_offset / minibatch_size);
        }

        clock_t end = clock();
        float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        
        if (test_samples != NULL) {
            float accuracy = convnn_evaluate(convnn, test_samples, test_count) * 100;
            printf("Epoch %d completed - Epoch time: %.2fs - Accuracy: %.2f%%\n", epoch, time_spent, accuracy);
        } else
            printf("Epoch %d completed.\n", epoch);

    }
    return convnn;
}