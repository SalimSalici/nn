#ifndef _CNN_H
#define _CNN_H

#include "nn.h"
#include "cmpl.h"

#define CMPL_MAX_LAYERS 32

typedef struct CNN {
    int group_count; // Basically number of samples in a minibatch (N)
    int num_cmpl_layers;
    Cmpl* cmpl_layers[CMPL_MAX_LAYERS];
    NN* nn;
} CNN;

CNN* cnn_malloc() {
    CNN* cnn = (CNN*)malloc(sizeof(CNN));
    for (int i = 0; i < CMPL_MAX_LAYERS; i++)
        cnn->cmpl_layers[i] = NULL;
    cnn->nn = NULL;
    cnn->num_cmpl_layers = 0;
    return cnn;
}

void cnn_free(CNN* cnn) {
    for (int i = 0; i < cnn->num_cmpl_layers; i++)
        cmpl_free(cnn->cmpl_layers[i]);
    nn_free(cnn->nn);
    free(cnn);
}

CNN* cnn_add_cmpl_layer(CNN* cnn, Cmpl* cmpl) {
    cnn->cmpl_layers[cnn->num_cmpl_layers] = cmpl;

    if (cnn->num_cmpl_layers == 0)
        cmpl->is_first = 1;
    else 
        cmpl->is_first = 0;

    cnn->num_cmpl_layers++;
    return cnn;
}

CNN* cnn_set_nn(CNN* cnn, NN* nn) {
    cnn->nn = nn;
    return cnn;
}

Mat* cnn_feedforward(CNN* cnn, Mat** inputs, int minibatch_count) {
    assert(cnn->num_cmpl_layers > 0);
    assert(cnn->group_count == minibatch_count);

    // Forward inputs through all convolutional maxpool layers
    Mat* cmpl_outputs = cmpl_forward(cnn->cmpl_layers[0], NULL, inputs);

    for (int i = 1; i < cnn->num_cmpl_layers; i++) {
        cmpl_outputs = cmpl_forward(cnn->cmpl_layers[i], cmpl_outputs, NULL);
    }

    Cmpl* last_cmpl = cnn->cmpl_layers[cnn->num_cmpl_layers - 1];

    // Prerare output of last convolutional maxpool layer to be in the right shape for ffnn
    mat_view(cmpl_outputs, last_cmpl->maxpool->n, last_cmpl->maxpool->h_out * last_cmpl->maxpool->w_out * last_cmpl->maxpool->c);
    mat_transpose(cmpl_outputs);
    
    // Forward through all fully connected layers
    Mat* nn_outputs = nn_feedforward(cnn->nn, cmpl_outputs);

    // Return to original shape of last convolutional maxpool layer
    mat_transpose(cmpl_outputs);
    mat_view(cmpl_outputs, last_cmpl->maxpool->n * last_cmpl->maxpool->h_out, last_cmpl->maxpool->w_out * last_cmpl->maxpool->c);

    return nn_outputs;
}

CNN* cnn_backprop(CNN* cnn, Mat* outputs, int inputs_count) {
    assert(inputs_count <= cnn->group_count);

    nn_backprop(cnn->nn, outputs);

    Mat* ds = layer_backward_for_cmpl(cnn->nn->layers[0], cnn->nn->layers[1]);

    for(int i = cnn->num_cmpl_layers - 1; i >= 0; i--) {
        ds = cmpl_backward(cnn->cmpl_layers[i], ds);
    }
    
    return cnn;
}

CNN* cnn_set_group_count(CNN* cnn, int n) {
    cnn->group_count = n;
    for (int i = 0; i < cnn->num_cmpl_layers; i++) {
        conv2d_set_n(cnn->cmpl_layers[i]->conv2d, n);
        maxpool_set_n(cnn->cmpl_layers[i]->maxpool, n);
    }
    nn_set_layers_group_count(cnn->nn, n);
    return cnn;
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

CNN* cnn_update_minibatch(CNN* cnn, float lr, float lambda, Sample* minibatch[], int minibatch_size, int training_count) {
    
    nn_set_mode(cnn->nn, NN_TRAINING);

    Mat* inputs[minibatch_size];
    for (int i = 0; i < minibatch_size; i++)
        inputs[i] = minibatch[i]->inputs;

    cnn_feedforward(cnn, inputs, minibatch_size);
    
    Mat* correct_outputs = cnn_merge_outputs(minibatch, minibatch_size);
    cnn_backprop(cnn, correct_outputs, minibatch_size);
    
    // Cmpl* last_cmpl = cnn->cmpl_layers[cnn->num_cmpl_layers - 1];
    // cmpl_update_weights_and_biases(last_cmpl, lr);

    for (int i = 0; i < cnn->num_cmpl_layers; i++) {
        cmpl_update_weights_and_biases(cnn->cmpl_layers[i], lr);
    }

    nn_update_weights_and_biases(cnn->nn, lr, lambda, training_count);

    mat_free(correct_outputs);

    return cnn;
}

float cnn_evaluate(CNN* cnn, Sample* test_samples[], int test_samples_count, float* loss) {

    NN* nn = cnn->nn;

    cnn_set_group_count(cnn, 1);
    nn_set_mode(nn, NN_INFERENCE);

    int correct_predictions = 0;

    if (loss)
        *loss = 0;

    for (int i = 0; i < test_samples_count; i++) {
        Mat* inputs = test_samples[i]->inputs;
        Mat* outputs = cnn_feedforward(cnn, &inputs, 1);
        Mat* correct_outputs = test_samples[i]->outputs;

        int prediction = nn_argmax(outputs->data, outputs->rows);
        int correct = nn_argmax(test_samples[i]->outputs->data, test_samples[i]->outputs->rows);

        if (prediction == correct)
            correct_predictions++;

        if (loss) {
            switch(nn->loss_function) {
                case NN_MSE_LOSS:
                    for (int i = 0; i < outputs->rows; i++) {
                        float diff = outputs->data[i] - correct_outputs->data[i];
                        *loss += diff*diff;
                    }
                    break;
                case NN_BCE_LOSS:
                    for (int i = 0; i < outputs->rows; i++) {
                        float y = correct_outputs->data[i];
                        float p =  outputs->data[i];
                        *loss -= y * log(p + 0.00000000001) + (1 - y) * log(1 - p + 0.00000000001);
                    }
                    break;
                case NN_NLL_LOSS:
                    *loss -= log(outputs->data[correct]);
                    break;
            }
        }
    }

    if (loss)
        *loss /= test_samples_count;

    return (float)correct_predictions / test_samples_count;
}

CNN* cnn_sgd(CNN* cnn, Sample* training_samples[], int training_samples_count, int epochs,
    int minibatch_size, float lr, float lambda, Sample* test_samples[], int test_samples_count) {

    printf("Starting cnn SGD. \nParameters: epochs=%d, minibatch_size=%d, lr=%f, lambda=%f\n", epochs, minibatch_size, lr, lambda);
    printf("Initial accuracy: %.2f%%\n\n", cnn_evaluate(cnn, test_samples, test_samples_count, NULL) * 100);

    clock_t begin_total = clock();
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        if (epoch != 0 && epoch % 30 == 0) {
            lr *= 0.1;
            printf("Learning rate updated. Current lr=%f\n", lr);
        }

        clock_t begin = clock();

        shuffle_pointers((void*)training_samples, training_samples_count);

        cnn_set_group_count(cnn, minibatch_size);

        for (int batch_offset = 0; batch_offset < training_samples_count; batch_offset += minibatch_size)
            cnn_update_minibatch(cnn, lr, lambda, training_samples + batch_offset, minibatch_size, training_samples_count);

        clock_t end = clock();
        float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        
        if (test_samples != NULL) {
            float accuracy = cnn_evaluate(cnn, test_samples, test_samples_count, NULL) * 100;
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

CNN* cnn_one_epoch(CNN* cnn, Sample* training_samples[], int training_samples_count, int epochs,
    int minibatch_size, float lr, float lambda, Sample* test_samples[], int test_samples_count) {

    float loss = 0;

    clock_t begin = clock();

    shuffle_pointers((void*)training_samples, training_samples_count);

    cnn_set_group_count(cnn, minibatch_size);

    for (int batch_offset = 0; batch_offset < training_samples_count; batch_offset += minibatch_size)
        cnn_update_minibatch(cnn, lr, lambda, training_samples + batch_offset, minibatch_size, training_samples_count);

    clock_t end = clock();
    float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    
    if (test_samples != NULL) {
        float accuracy = cnn_evaluate(cnn, test_samples, test_samples_count, &loss) * 100;
        printf("Epoch completed - Epoch time: %.2fs - Loss: %f - Accuracy: %.2f%%\n", time_spent, loss, accuracy);
    } else
        printf("Epoch completed.\n");

    return cnn;
}

#endif