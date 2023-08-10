#ifndef _CNN_H
#define _CNN_H

#include <string.h>
#include "mat.h"
#include "cpl.h"
#include "nn.h"

typedef struct CNN {
    CPL* cpl;
    NN* nn;
    int group_count;
} CNN;

CNN* cnn_malloc(CPL* cpl, NN* nn, int group_count) {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));
    cnn->cpl = cpl;
    cnn->nn = nn;
    cnn->group_count = group_count;
    return cnn;
}

void cnn_free(CNN* cnn) {
    cpl_free(cnn->cpl);
    nn_free(cnn->nn);
}

Mat* cnn_merge_maxpools_for_nn(CNN* cnn, int inputs_count) {
    assert(inputs_count <= cnn->group_count);

    CPL* cpl = cnn->cpl;

    Mat* res = mat_malloc(inputs_count, cpl->maxpoolmap_side * cpl->maxpoolmap_side * cpl->feature_count);
    float* cur_res = res->data;

    for (int i = 0; i < inputs_count; i++) {
        memcpy(cur_res + i * res->down, cpl->maxpoolmaps_a[i]->data, sizeof(float) * res->cols);
    }
    if (inputs_count != 1)
        return mat_transpose(res);
    else {
        int tmp = res->rows;
        res->rows = res->cols;
        res->cols = tmp;
        res->right = 1;
        res->down = res->cols;
        return res;
    }
}

Mat* cnn_merge_outputs(Sample* samples[], int samples_count) {
    int output_rows = samples[0]->outputs->rows;
    Mat* res_outputs = mat_malloc(samples_count, output_rows);
    for (int i = 0; i < samples_count; i++) {
        memcpy(res_outputs->data + i * output_rows, samples[i]->outputs->data, sizeof(float) * output_rows);
    }
    mat_transpose(res_outputs);
    return res_outputs;
}

Mat* cnn_feedforward(CNN* cnn, Mat* inputs[], int inputs_count) {
    assert(inputs_count <= cnn->group_count);

    cpl_forward(cnn->cpl, inputs, inputs_count);

    Mat* nn_inputs = cnn_merge_maxpools_for_nn(cnn, inputs_count);

    Mat* nn_output = nn_feedforward(cnn->nn, nn_inputs);
    mat_free(nn_inputs);

    return nn_output;
}

CNN* cnn_backprop(CNN* cnn, Mat* outputs, int inputs_count) {
    assert(inputs_count <= cnn->group_count);

    nn_backprop(cnn->nn, outputs);
    layer_backward(cnn->nn->layers[0], cnn->nn->layers[1]);
    cpl_backward(cnn->cpl, cnn->nn->layers[0]->d, inputs_count);
    
    return cnn;
}

CNN* cnn_update_weights_and_biases(CNN* cnn, float lr, float lambda, int minibatch_size, int training_count) {
    nn_update_weights_and_biases(cnn->nn, lr, lambda, training_count);
    cpl_update_weights_and_biases(cnn->cpl, lr, lambda, training_count, minibatch_size);
    return cnn;
}

CNN* cnn_update_minibatch(CNN* cnn, float lr, float lambda, Sample* minibatch[], int minibatch_size, int training_count) {
    
    nn_set_mode(cnn->nn, NN_TRAINING);

    Mat* inputs[minibatch_size];
    for (int i = 0; i < minibatch_size; i++)
        inputs[i] = minibatch[i]->inputs;

    cnn_feedforward(cnn, inputs, minibatch_size);
    Mat* correct_outputs = cnn_merge_outputs(minibatch, minibatch_size);
    cnn_backprop(cnn, correct_outputs, minibatch_size);
    cnn_update_weights_and_biases(cnn, lr, lambda, minibatch_size, training_count);

    mat_free(correct_outputs);

    return cnn;
}

float cnn_evaluate(CNN* cnn, Sample* test_samples[], int test_samples_count) {

    NN* nn = cnn->nn;

    nn_set_layers_group_count(nn, 1);
    nn_set_mode(nn, NN_INFERENCE);

    int correct_predictions = 0;

    for (int i = 0; i < test_samples_count; i++) {
        Mat* inputs = test_samples[i]->inputs;
        Mat* outputs = cnn_feedforward(cnn, &inputs, 1);

        int prediction = nn_argmax(outputs->data, outputs->rows);
        int correct = nn_argmax(test_samples[i]->outputs->data, test_samples[i]->outputs->rows);

        if (prediction == correct)
            correct_predictions++;

        // mat_free(outputs);
    }

    return (float)correct_predictions / test_samples_count;
}

CNN* cnn_sgd(CNN* cnn, Sample* training_samples[], int training_samples_count, int epochs,
    int minibatch_size, float lr, float lambda, Sample* test_samples[], int test_samples_count) {

    printf("Starting cnn SGD. \nParameters: epochs=%d, minibatch_size=%d, lr=%f, lambda=%f\n", epochs, minibatch_size, lr, lambda);

    printf("Initial accuracy: %.2f%%\n\n", cnn_evaluate(cnn, test_samples, test_samples_count) * 100);

    clock_t begin_total = clock();
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        if (epoch != 0 && epoch % 100 == 0) {
            lr *= 0.1;
            printf("Learning rate updated. Current lr=%f\n", lr);
        }

        clock_t begin = clock();

        shuffle_pointers((void*)training_samples, training_samples_count);

        nn_set_layers_group_count(cnn->nn, minibatch_size);

        for (int batch_offset = 0; batch_offset < training_samples_count; batch_offset += minibatch_size)
            cnn_update_minibatch(cnn, lr, lambda, training_samples + batch_offset, minibatch_size, training_samples_count);

        clock_t end = clock();
        float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        
        if (test_samples != NULL) {
            float accuracy = cnn_evaluate(cnn, test_samples, test_samples_count) * 100;
            printf("Epoch %d completed - Epoch time: %.2fs - Accuracy: %.2f%%\n", epoch, time_spent, accuracy);
        } else
            printf("Epoch %d completed.\n", epoch);

    }

    clock_t end_total = clock();
    float time_spent_total = (float)(end_total - begin_total) / CLOCKS_PER_SEC;
    printf("Training completed. Total time: %.2fs\n", time_spent_total);
    printf("Parameters: epochs=%d, minibatch_size=%d, lr=%f, lambda=%f\n", epochs, minibatch_size, lr, lambda);

    return cnn;
}

CNN* cnn_im2col_samples(CNN* cnn, Sample* samples[], int samples_count) {

    CPL* cpl = cnn->cpl;
    for (int i = 0; i < samples_count; i++) {
        Mat* new_inputs = mat_malloc(cpl->input_rows, cpl->input_cols);
        new_inputs = cpl_im2col(new_inputs, samples[i]->inputs, cpl->kernel_side, cpl->kernel_stride);
        samples[i]->inputs = new_inputs;
    }

    return cnn;
}

#endif