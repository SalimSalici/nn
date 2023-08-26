#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "mat.h"
#include "nn.h"
#include "sample.h"
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"

#include "tensor.h"
#include "mec.h"
#include "conv2d.h"
#include "maxpool.h"

int main(int argc, char* argv[]) {

    srand(time(NULL) * time(NULL));

    goto_set_num_threads(1);
    openblas_set_num_threads(1);

    // float data[49 * 2] = {
    //     0, 0, 0, 0, 0, 0, 0,
    //     0, 2, 2, 1, 1, 2, 0,
    //     0, 2, 0, 1, 1, 0, 0,
    //     0, 2, 0, 1, 2, 0, 0,
    //     0, 1, 1, 1, 1, 1, 0,
    //     0, 0, 0, 1, 0, 2, 0,
    //     0, 0, 0, 0, 0, 0, 0
    // };

    // float kernel[9][1] = {{1}, {0}, {0}, {1}, {1}, {1}, {1}, {0}, {-1}};
    // Mat* ker_mat = mat_malloc_cpy(9, 1, kernel);

    // Tensor* tens = tensor_malloc_nodata(1, 7, 7, 1);
    // tensor_attach_data(tens, data);

    // Tensor* im2mecced = im2mec_input(tens, 3, 3, 1);
    // Tensor* res = mec_conv(im2mecced, ker_mat, 3, 1);

    // mat_print(res->mat);
    // printf("\n");

    // Conv2d* conv2d = conv2d_malloc(2, 3, 3, 3, 2, 2, 1, 2);

    // float data2[3*3*3 * 2] = {
    //     1, 2, 3,        4, 5, 6,        7, 8, 9,
    //     10, 11, 12,     13, 14, 15,     16, 17, 18,
    //     19, 20, 21,     22, 23, 24,     25, 26, 27,
    //     1, 2, 3,        4, 5, 6,        7, 8, 9,
    //     10, 11, 12,     13, 14, 15,     16, 17, 18,
    //     19, 20, 21,     22, 23, 24,     25, 26, 27
    // };

    // // float kernel2[12][1] = {{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}};
    // // Mat* ker_mat2 = mat_malloc_cpy(12, 1, kernel2);
    // Mat* ker_mat2 = mat_malloc(12, 2);
    // mat_fill_func(ker_mat2, ker_mat2, mat_standard_norm_filler_cb, NULL);

    // Tensor* tens2 = tensor_malloc_nodata(2, 3, 3, 3);
    // tensor_attach_data(tens2, data2);
    // mat_fill_func(tens2->mat, tens2->mat, mat_standard_norm_filler_cb, NULL);

    // Tensor* im2mecced2 = im2mec_input(tens2, 2, 2, 1);

    // mat_print(tens2->mat);
    // printf("\n");
    // mat_print(im2mecced2->mat);
    // printf("\n");

    // conv2d_lower(conv2d, tens2->mat);
    // mat_print(conv2d->inputs_lowered);

    // printf("--------------\n\n");

    // Tensor* res2 = mec_conv(im2mecced2, ker_mat2, 2, 1);

    // tensor_view_mat(res2, res2->dim0, res2->dim1 * res2->dim2 * res2->dim3);

    // mat_print(res2->mat);
    // printf("\n");

    // float biases[1][2] = {{3, 2}};

    // mat_free(conv2d->kernels);
    // mat_free(conv2d->biases);
    // conv2d->kernels = ker_mat2;
    // conv2d->biases = mat_malloc_cpy(1, 2, biases);
    // // conv2d_bias_forward(conv2d);
    // conv2d_forward(conv2d, tens2->mat);
    // mat_print(conv2d->conv_hnwc_z);
    // printf("\n");
    // mat_print(conv2d->conv_hnwc_a);

    // printf("--------------\n");

    // Maxpool* maxpool = maxpool_malloc(2, 2, 2, 2, 2, 2, 1);

    // maxpool_forward(maxpool, conv2d->conv_hnwc_z);

    // mat_print(maxpool->outputs);


    Mat* input = mat_malloc(10 * 28, 28 * 1);
    mat_fill_func(input, input, mat_standard_norm_filler_cb, NULL);
    
    Conv2d* conv2d = conv2d_malloc(10, 28, 28, 1, 5, 5, 1, 40);
    Maxpool* maxpool = maxpool_malloc(conv2d->n, conv2d->h_out, conv2d->w_out, conv2d->c_out, 2, 2, 2);

    printf("ciao");
    conv2d_forward(conv2d, input);
    maxpool_forward(maxpool, conv2d->conv_hnwc_z);

    // mat_print(input);
    // printf("\n");
    // mat_print(conv2d->inputs_lowered);
    // printf("\n");
    mat_print(conv2d->conv_hnwc_z);
    printf("\n");
    mat_print(maxpool->outputs);
    printf("\n");

    int* cur_i = maxpool->stored_indeces;
    for (int r = 0; r < maxpool->outputs->rows; r++) {
        printf("{");
        for (int c = 0; c < maxpool->outputs->cols; c++) {
            printf("%d - %f - %f | ", *cur_i, maxpool->outputs->data[cur_i - maxpool->stored_indeces], conv2d->conv_hnwc_z->data[*cur_i]);
            cur_i++;
        }
        printf("}\n");
    }

    printf("%d, %d\n", conv2d->conv_hnwc_z->rows, conv2d->conv_hnwc_z->cols);
    printf("conv2d->n, conv2d->h_out, conv2d->w_out, conv2d->c_out: %d, %d, %d, %d\n", conv2d->n, conv2d->h_out, conv2d->w_out, conv2d->c_out);
    

    return 0;

    // Tensor* next_grad = tensor_malloc(res2->dim0, res2->dim1, res2->dim2, res2->dim3);
    // mat_fill(next_grad->mat, 1.0);

    // Mat* grad = mec_conv_backwards_kernel(im2mecced2, next_grad, 2, 2, 3, 1);

    // Tensor* grad2 = mec_conv_backwards_prev(ker_mat2, next_grad, 2, 2, 3, 1, 3);

    // mat_print(grad);
    // printf("\n");
    // mat_print(grad2->mat);
    // printf("\n");
    // Tensor* grad_img = mec2im_input_additive(grad2, 2, 2, 1);
    // mat_print(tens2->mat);
    // printf("\n");
    // mat_print(grad_img->mat);

    // printf("\nCIAO");

    return 0;

    NN* nn = nn_malloc(NN_NLL_LOSS);
    nn_add_layer(nn, layer_malloc(0, 28*28, NN_RELU_ACT, 0.2));
    nn_add_layer(nn, layer_malloc(28*28, 1000, NN_RELU_ACT, 0.5));
    nn_add_layer(nn, layer_malloc(1000, 10, NN_SOFTMAX_ACT, 0.5));
    nn_initialize_xavier(nn);

    float lr = 0.1; // learning rate
    float lambda = 0.0; // L2 regularization
    int epochs = 30;
    int minibatch_size = 50;
    int training_samples_count = 60000;
    int test_samples_count = 10000;
    float black = 0;
    float white = 1;

    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);
    // MnistSample* training_data = mnist_load_samples("data/fashion/train-images.idx3-ubyte", "data/fashion/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    // MnistSample* test_data = mnist_load_samples("data/fashion/t10k-images.idx3-ubyte", "data/fashion/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    Sample** training_samples = mnist_samples_to_samples(training_data, training_samples_count, black, white);
    Sample** test_samples = mnist_samples_to_samples(test_data, test_samples_count, black, white);

    nn_sgd(nn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);

    return 0;
}