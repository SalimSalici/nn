#ifndef _NN_H
#define _NN_H

#include <assert.h>
#include <string.h>
#include <math.h>
#include "helper.h"
#include "mat.h"
#include "sample.h"
#include "openblas_config.h"
#include "cblas.h"

#define NN_MAX_LAYERS 32

#define NN_SIGMOID(z) (1 / (1 + exp(-z)))
#define NN_SIGMOID_PRIME(z) (NN_SIGMOID(z) * (1 - NN_SIGMOID(z)))

typedef enum layer_activation_t {
    NN_SIGMOID_ACT,
    NN_RELU_ACT,
    NN_TANH_ACT,
    NN_SOFTMAX_ACT,
    NN_NONE_ACT
} layer_activation_t;

typedef enum loss_function_t {
    NN_MSE_LOSS, // Means squared error
    NN_BCE_LOSS, // Binary Cross-Entropy
    NN_NLL_LOSS // Negative log-likelyhood
} loss_function_t;

typedef enum layer_mode_t {
    NN_INFERENCE,
    NN_TRAINING
} layer_mode_t;

typedef struct Layer {
    int in_count; // Number of units in the previous layer
    int out_count; // Number of units in this layer
    layer_activation_t activation; // Activation function of this layer
    layer_mode_t mode; // Activation function of this layer
    int group_count; // Activation function of this layer
    float dropout_rate; // Activation function of this layer
    int is_last;
    Mat* w; // Weights that connect the previous layer to this layer 
    Mat* b; // Biases of this layer
    Mat* z; // Unit values before activation
    Mat* z_prime; // Unit values before activation
    Mat* a; // Unit values after activation
    Mat* d; // Unit errors
    Mat* loss_g; // Gradient of loss function with respect to output activations: dJ/da
    Mat* dropout_mask;
} Layer;

typedef struct NN {
    int num_layers;
    loss_function_t loss_function;
    Layer* layers[NN_MAX_LAYERS];
} NN;

float nn_mat_sigmoid_cb(float cur, int row, int col, void* func_args) {
    return 1 / (1 + exp(-cur));
}

float nn_mat_sigmoid_prime_cb(float cur, int row, int col, void* func_args) {
    float sigm = 1 / (1 + exp(-cur));
    return sigm * (1 - sigm);
}

float nn_mat_tanh_cb(float cur, int row, int col, void* func_args) {
    return tanh(cur);
}

float nn_mat_tanh_prime_cb(float cur, int row, int col, void* func_args) {
    float th = tanh(cur);
    return 1 - th * th;
}

float nn_mat_relu_cb(float cur, int row, int col, void* func_args) {
    return cur > 0.0 ? cur : 0.0;
}

float nn_mat_relu_prime_cb(float cur, int row, int col, void* func_args) {
    return cur > 0.0 ? 1 : 0.0;
}

Layer* layer_malloc(int in_count, int out_count, layer_activation_t activation, float dropout_rate) {
    assert(in_count >= 0 && out_count >= 0);
    
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->in_count = in_count;
    layer->out_count = out_count;
    layer->activation = activation;
    layer->dropout_rate = dropout_rate;
    layer->mode = NN_INFERENCE;
    layer->group_count = 1;
    layer->w = mat_malloc(out_count, in_count);
    layer->b = mat_malloc(out_count, 1);
    layer->z = mat_malloc(out_count, 1);
    layer->z_prime = mat_malloc(out_count, 1);
    layer->a = mat_malloc(out_count, 1);
    layer->d = mat_malloc(out_count, 1);
    layer->loss_g = mat_malloc(out_count, 1);
    layer->dropout_mask = mat_malloc(out_count, 1);
    return layer;
}

void layer_free(Layer* layer) {
    mat_free(layer->w);
    mat_free(layer->b);
    mat_free(layer->z);
    mat_free(layer->z_prime);
    mat_free(layer->a);
    mat_free(layer->d);
    mat_free(layer->loss_g);
    mat_free(layer->dropout_mask);
    free(layer);
}

Layer* layer_prepare_matrices(Layer* layer, int group_count) {
    mat_free(layer->z);
    mat_free(layer->z_prime);
    mat_free(layer->a);
    mat_free(layer->d);
    mat_free(layer->loss_g);
    mat_free(layer->dropout_mask);
    layer->group_count = group_count;
    int out_count = layer->out_count;
    layer->z = mat_malloc(out_count, group_count);
    layer->z_prime = mat_malloc(out_count, group_count);
    layer->a = mat_malloc(out_count, group_count);
    layer->d = mat_malloc(out_count, group_count);
    layer->loss_g = mat_malloc(out_count, group_count);
    layer->dropout_mask = mat_malloc(out_count, group_count);
    return layer;
}

Layer* layer_initialize_zero(Layer* layer) {
    mat_fill_func(layer->w, layer->w, mat_zero_filler_cb, NULL);
    mat_fill(layer->b, 0.0);
    return layer;
}

Layer* layer_initialize_xavier(Layer* layer) {
    float* norm_args[2];
    float mean = 0.0;
    float sd = 1 / sqrt(layer->in_count);
    norm_args[0] = &mean;
    norm_args[1] = &sd;
    mat_fill_func(layer->w, layer->w, mat_norm_filler_cb, norm_args);
    mat_fill(layer->b, 0.0);
    return layer;
}

Layer* layer_initialize_normal(Layer* layer, float mean, float sd) {
    float* norm_args[2];
    norm_args[0] = &mean;
    norm_args[1] = &sd;
    mat_fill_func(layer->w, layer->w, mat_norm_filler_cb, norm_args);
    mat_fill(layer->b, 0.0);
    return layer;
}

Layer* layer_initialize_standard(Layer* layer) {
    mat_fill_func(layer->w, layer->w, mat_standard_norm_filler_cb, NULL);
    mat_fill_func(layer->b, layer->b, mat_standard_norm_filler_cb, NULL);
    return layer;
}

Mat* layer_apply_softmax(Layer* layer) {
    for (int c = 0; c < layer->z->cols; c++) {
        float* cur_z = layer->z->data + c;
        float max = -INFINITY;

        for (int r = 0; r < layer->z->rows; r++) {
            if (*cur_z > max) max = *cur_z;
            cur_z += layer->z->cols;
        }

        cur_z = layer->z->data + c;
        float den = 0;
        for (int r = 0; r < layer->z->rows; r++) {
            den += exp(*cur_z - max);
            cur_z += layer->z->cols;
        }

        cur_z = layer->z->data + c;
        float* cur_a = layer->a->data + c;;
        for (int r = 0; r < layer->z->rows; r++) {
            *cur_a = exp(*cur_z - max) / den;
            cur_z += layer->z->cols;
            cur_a += layer->a->cols;
        }
    }

    return layer->a;
}

Layer* layer_add_bias_to_z(Layer* layer) {
    if (layer->group_count == 1) {
        mat_add(layer->z, layer->z, layer->b);
    } else {
        int z_rows = layer->z->rows;
        int z_cols = layer->z->cols;
        float* cur_z = layer->z->data;
        float* b = layer->b->data;
        for (int r = 0; r < z_rows; r++) {
            for (int c = 0; c < z_cols; c++) {
                *cur_z += b[r];
                cur_z++;
            }
        }
    }
    return layer;
}

Layer* layer_generate_dropout_mask(Layer* layer) {

    float dropout_rate = layer->dropout_rate;
    float* data = layer->dropout_mask->data;
    for (int i = 0; i < layer->dropout_mask->rows * layer->dropout_mask->cols; i++) {
        if ((float)rand() / RAND_MAX < dropout_rate)
            data[i] = 0.0;
        else 
            data[i] = 1.0;
    }

    return layer;
}

Mat* layer_forward_first(Layer* layer, Mat* inputs) {
    
    mat_free(layer->a);
    
    if (layer->mode == NN_TRAINING ) {
        if (layer->dropout_rate != 0.0) {
            layer_generate_dropout_mask(layer);
            layer->a = mat_hadamard_prod(NULL, inputs, layer->dropout_mask, 1);
        } else {
            layer->a = mat_cpy(inputs);
        }
    } else if (layer->mode == NN_INFERENCE) {
        layer->a = mat_cpy(inputs);
        if (layer->dropout_rate != 0.0)
            layer->a = mat_scale(layer->a, layer->a, 1 - layer->dropout_rate);
    }

    return layer->a;
}

Mat* layer_forward(Layer* layer, Mat* inputs) {
    if (!inputs->t)
        assert(layer->group_count == inputs->cols);
    else 
        assert(layer->group_count == inputs->rows);

    if (layer->group_count == 1)
        mat_mult_mv(layer->z, layer->w, inputs, 1.0, 0.0);
    else
        mat_mult_mm(layer->z, layer->w, inputs, 1.0, 0.0);

    layer_add_bias_to_z(layer);

    switch(layer->activation) {
        case NN_SIGMOID_ACT:
            mat_fill_func(layer->a, layer->z, nn_mat_sigmoid_cb, NULL);
            break;
        case NN_SOFTMAX_ACT:
            layer_apply_softmax(layer);
            break;
        case NN_RELU_ACT:
            mat_fill_func(layer->a, layer->z, nn_mat_relu_cb, NULL);
            break;
        case NN_TANH_ACT:
            mat_fill_func(layer->a, layer->z, nn_mat_tanh_cb, NULL);
            break;
        case NN_NONE_ACT:
            break;
    }

    if (!layer->is_last) {
        if (layer->mode == NN_TRAINING) {
            if (layer->dropout_rate != 0.0) {
                layer_generate_dropout_mask(layer);
                layer->a = mat_hadamard_prod(layer->a, layer->a, layer->dropout_mask, 1);
            }
        } else if (layer->mode == NN_INFERENCE) {
            if (layer->dropout_rate != 0.0)
                layer->a = mat_scale(layer->a, layer->a, 1 - layer->dropout_rate);
        }
    }

    return layer->a;
}

Layer* layer_calculate_loss_gradient(Layer* layer, Mat* outputs, loss_function_t loss_function) {
    switch(loss_function) {
        case NN_MSE_LOSS:
            mat_sub(layer->loss_g, layer->a, outputs);
            break;
        case NN_BCE_LOSS:
            mat_sub(layer->loss_g, layer->a, outputs);
            mat_fill_func(layer->z_prime, layer->z, nn_mat_sigmoid_prime_cb, NULL);
            mat_hadamard_div(layer->loss_g, layer->loss_g, layer->z_prime, 1);
            break;
        case NN_NLL_LOSS:
            mat_hadamard_div(layer->loss_g, outputs, layer->a, -1);
            break;
    }
    return layer;
}

Mat* layer_backward_last_mse_sigmoid(Layer* layer, Mat* outputs) {
    layer->d = mat_sub(layer->d, layer->a, outputs);
    layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_sigmoid_prime_cb, NULL);
    layer->d = mat_hadamard_prod(layer->d, layer->d, layer->z_prime, 1);
    return layer->d;
}

Mat* layer_backward_last_ce(Layer* layer, Mat* outputs) {
    layer->d = mat_sub(layer->d, layer->a, outputs);
    return layer->d;
}

void layer_softmax_jacobians(int count, Mat* jacobians[count],  Mat* a) {
    assert(a->cols == count);
    for (int i = 0; i < count; i++) {
        float* a_data = a->data + i /* * a->right ... se a fosse trasposta sarebbe da fare così*/;
        int a_down = a->down;
        Mat* jacobian = mat_malloc(a->rows, a->rows);
        for (int r = 0; r < a->rows; r++) {
            for (int c = r; c < a->rows; c++) {
                if (r == c) jacobian->data[c + r * a->rows] = (*(a_data + r * a_down)) * (1 - *(a_data + c * a_down));
                else {
                    jacobian->data[c + r * a->rows] = -(*(a_data + r * a_down)) * (*(a_data + c * a_down));
                    jacobian->data[r + c * a->rows] = -(*(a_data + r * a_down)) * (*(a_data + c * a_down));
                }
            }
        }
        jacobians[i] = jacobian;
    }
}

Mat* layer_backward_last(Layer* layer, Mat* outputs, loss_function_t loss_function) {

    if (layer->activation == NN_SIGMOID_ACT && loss_function == NN_BCE_LOSS)
        return layer_backward_last_ce(layer, outputs);

    if (layer->activation == NN_SOFTMAX_ACT && loss_function == NN_NLL_LOSS)
        return layer_backward_last_ce(layer, outputs);

    layer_calculate_loss_gradient(layer, outputs, loss_function);

    switch(layer->activation) {
        case NN_SIGMOID_ACT:
            layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_sigmoid_prime_cb, NULL);
            layer->d = mat_hadamard_prod(layer->d, layer->loss_g, layer->z_prime, 1);
            break;
        case NN_RELU_ACT:
            layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_relu_prime_cb, NULL);
            layer->d = mat_hadamard_prod(layer->d, layer->loss_g, layer->z_prime, 1);
            break;
        case NN_TANH_ACT:
            layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_tanh_prime_cb, NULL);
            layer->d = mat_hadamard_prod(layer->d, layer->loss_g, layer->z_prime, 1);
            break;
        case NN_SOFTMAX_ACT:
            int number_of_jacobians = layer->group_count;
            Mat** jacobians = (Mat**)malloc(sizeof(Mat*) * number_of_jacobians);
            layer_softmax_jacobians(number_of_jacobians, jacobians, layer->a);
            for (int i = 0; i < number_of_jacobians; i++) {
                Mat* jacobian = jacobians[i];
                cblas_sgemv(CblasRowMajor, 
                    CblasTrans,
                    jacobian->rows, jacobian->cols,
                    1.0, jacobian->data, jacobian->cols, layer->loss_g->data + i, layer->loss_g->cols, 
                    0.0, layer->d->data + i, layer->d->cols);

                mat_free(jacobians[i]);
            }
            free(jacobians);
            break;
        case NN_NONE_ACT:
            break;
    }

    return layer->d;
}

Mat* layer_backward(Layer* layer, Layer* next_layer) {
    layer->d = mat_mult_mm(layer->d, mat_transpose(next_layer->w), next_layer->d, 1.0, 0.0);
    mat_transpose(next_layer->w);

    switch(layer->activation) {
        case NN_SIGMOID_ACT:
            layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_sigmoid_prime_cb, NULL);
            break;
        case NN_TANH_ACT:
            layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_tanh_prime_cb, NULL);
            break;
        case NN_RELU_ACT:
            layer->z_prime = mat_fill_func(layer->z_prime, layer->z, nn_mat_relu_prime_cb, NULL);
            break;
        case NN_SOFTMAX_ACT:
            printf("This frameworks doesn't support a softmax layer in the hidden layers.");
            exit(0);
            break;
        case NN_NONE_ACT:
            break;
    }

    if (layer->activation != NN_NONE_ACT)
        layer->d = mat_hadamard_prod(layer->d, layer->d, layer->z_prime, 1);

    if (!layer->is_last && layer->dropout_rate != 0) {
        layer->d = mat_hadamard_prod(layer->d, layer->d, layer->dropout_mask, 1);
    }

    return layer->d;
}

Mat* layer_backward_for_cmpl(Layer* layer, Layer* next_layer) {

    mat_view(layer->d, next_layer->d->cols, next_layer->w->cols);

    layer->d = mat_mult_mm(layer->d, mat_transpose(next_layer->d), next_layer->w, 1.0, 0.0);
    mat_transpose(next_layer->d);

    if (!layer->is_last && layer->dropout_rate != 0) {
        layer->d = mat_hadamard_prod(layer->d, layer->d, mat_transpose(layer->dropout_mask), 1);
        mat_transpose(layer->dropout_mask);
    }

    return layer->d;
}

Layer* layer_update_weights_and_biases(Layer* layer, Layer* prev_layer, float lr, float lambda, int training_count) {

    float reg_factor = lr * lambda / (float)training_count;

    layer->w = mat_mult_mm(layer->w, layer->d, mat_transpose(prev_layer->a), -lr / (float)layer->group_count, 1.0 - reg_factor);
    mat_transpose(prev_layer->a);
    
    Mat* ones = mat_malloc(layer->group_count, 1);
    ones = mat_fill(ones, 1.0);
    layer->b = mat_mult_mv(layer->b, layer->d, ones, -lr / (float)layer->group_count, 1.0);
    mat_free(ones);
    
    return layer;
}

NN* nn_malloc(loss_function_t loss_function) {
    NN* nn = (NN*)malloc(sizeof(NN));
    nn->num_layers = 0;
    nn->loss_function = loss_function;
    return nn;
}

void nn_free(NN* nn) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_free(nn->layers[i]);
    free(nn);
}

NN* nn_add_layer(NN* nn, Layer* l) {
    nn->layers[nn->num_layers] = l;
    l->is_last = 1;

    if (nn->num_layers > 0)
        nn->layers[nn->num_layers - 1]->is_last = 0;

    nn->num_layers++;
    return nn;
}

NN* nn_set_layers_group_count(NN* nn, int group_count) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_prepare_matrices(nn->layers[i], group_count);
    return nn;
}

NN* nn_initialize_zero(NN* nn) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_initialize_zero(nn->layers[i]);
    return nn;
}

NN* nn_initialize_xavier(NN* nn) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_initialize_xavier(nn->layers[i]);
    return nn;
}

NN* nn_initialize_normal(NN* nn, float mean, float sd) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_initialize_normal(nn->layers[i], mean, sd);
    return nn;
}

NN* nn_initialize_standard(NN* nn) {
    for (int i = 0; i < nn->num_layers; i++)
        layer_initialize_standard(nn->layers[i]);
    return nn;
}

NN* nn_set_mode(NN* nn, layer_mode_t mode) {
    Layer** layers = nn->layers;
    for (int i = 0; i < nn->num_layers; i++) {
        layers[i]->mode = mode;
    }
    return nn;
}

Mat* nn_feedforward(NN* nn, Mat* inputs) {
    layer_forward_first(nn->layers[0], inputs);
    for (int i = 1; i < nn->num_layers; i++) {
        Layer* l = nn->layers[i];
        Layer* l_prev = nn->layers[i-1];
        layer_forward(l, l_prev->a);
    }
    return nn->layers[nn->num_layers - 1]->a;
}

NN* nn_backprop(NN* nn, Mat* outputs) {
    int last_layer_idx = nn->num_layers - 1;
    layer_backward_last(nn->layers[last_layer_idx], outputs, nn->loss_function);

    for (int i = last_layer_idx - 1; i > 0; i--)
        layer_backward(nn->layers[i], nn->layers[i+1]);

    return nn;
}

void nn_print_weights(NN* nn) {
    for (int i = 1; i < nn->num_layers; i++) {
        mat_print(nn->layers[i]->w);
        printf("\n");
    }
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

void nn_merge_minibatch(Sample* minibatch[], int minibatch_size, Mat** inputs, Mat** outputs) {
    int input_rows = minibatch[0]->inputs->rows;
    int output_rows = minibatch[0]->outputs->rows;
    Mat* res_inputs = mat_malloc(minibatch_size, input_rows);
    Mat* res_outputs = mat_malloc(minibatch_size, output_rows);
    for (int i = 0; i < minibatch_size; i++) {
        memcpy(res_inputs->data + i * input_rows, minibatch[i]->inputs->data, sizeof(float) * input_rows);
        memcpy(res_outputs->data + i * output_rows, minibatch[i]->outputs->data, sizeof(float) * output_rows);
    }
    mat_transpose(res_inputs);
    mat_transpose(res_outputs);
    *inputs = res_inputs;
    *outputs = res_outputs;
}

NN* nn_update_weights_and_biases(NN* nn, float lr, float lambda, int training_count) {
    for (int i = 1; i < nn->num_layers; i++)
        layer_update_weights_and_biases(nn->layers[i], nn->layers[i-1], lr, lambda, training_count);
    return nn;
}

NN* nn_update_minibatch(NN* nn, float lr, float lambda, Sample* minibatch[], int minibatch_size, int training_count) {

    Mat* inputs;
    Mat* outputs;

    nn_set_mode(nn, NN_TRAINING);

    nn_merge_minibatch(minibatch, minibatch_size, &inputs, &outputs);

    nn_feedforward(nn, inputs);
    nn_backprop(nn, outputs);

    mat_free(inputs);
    mat_free(outputs);

    nn_update_weights_and_biases(nn, lr, lambda, training_count);

    return nn;
}

float nn_evaluate(NN* nn, Sample* test_samples[], int test_samples_count, float* loss) {

    nn_set_layers_group_count(nn, 1);

    nn_set_mode(nn, NN_INFERENCE);

    int correct_predictions = 0;

    if (loss)
        *loss = 0;

    for (int i = 0; i < test_samples_count; i++) {
        Mat* nn_inputs = test_samples[i]->inputs;
        Mat* nn_outputs = nn_feedforward(nn, nn_inputs);

        Mat* correct_outputs = test_samples[i]->outputs;

        int prediction = nn_argmax(nn_outputs->data, nn_outputs->rows);
        int correct = nn_argmax(correct_outputs->data, correct_outputs->rows);

        if (loss) {
            switch(nn->loss_function) {
                case NN_MSE_LOSS:
                    for (int i = 0; i < nn_outputs->rows; i++) {
                        float diff = nn_outputs->data[i] - correct_outputs->data[i];
                        *loss += diff*diff;
                    }
                    break;
                case NN_BCE_LOSS:
                    for (int i = 0; i < nn_outputs->rows; i++) {
                        float y = correct_outputs->data[i];
                        float p =  nn_outputs->data[i];
                        *loss -= y * log(p + 0.00000000001) + (1 - y) * log(1 - p + 0.00000000001);
                    }
                    break;
                case NN_NLL_LOSS:
                    *loss -= log(nn_outputs->data[correct]);
                    break;
            }
        }

        if (prediction == correct)
            correct_predictions++;
    }

    *loss /= test_samples_count;

    return (float)correct_predictions / test_samples_count;
}

NN* nn_sgd(NN* nn, Sample* training_samples[], int training_samples_count, int epochs,
    int minibatch_size, float lr, float lambda, Sample* test_samples[], int test_samples_count) {

    float loss;
    printf("Starting SGD. Initial accuracy: %.2f%%\n", nn_evaluate(nn, test_samples, test_samples_count, &loss) * 100);
    printf("Initial loss: %.4f\n", loss);

    clock_t begin_total = clock();
    
    for (int epoch = 0; epoch < epochs; epoch++) {

        clock_t begin = clock();

        shuffle_pointers((void*)training_samples, training_samples_count);

        nn_set_layers_group_count(nn, minibatch_size);

        for (int batch_offset = 0; batch_offset < training_samples_count; batch_offset += minibatch_size)
            nn_update_minibatch(nn, lr, lambda, training_samples + batch_offset, minibatch_size, training_samples_count);

        clock_t end = clock();
        float time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
        
        if (test_samples != NULL) {
            float accuracy = nn_evaluate(nn, test_samples, test_samples_count, &loss) * 100;
            printf("Epoch %d completed - Epoch time: %.2fs - Loss: %.20f - Accuracy: %.2f%%\n", epoch, time_spent, loss, accuracy);
        } else
            printf("Epoch %d completed.\n", epoch);

    }

    clock_t end_total = clock();
    float time_spent_total = (float)(end_total - begin_total) / CLOCKS_PER_SEC;
    printf("Training completed. Total time: %.2fs", time_spent_total);

    return nn;
}

#endif