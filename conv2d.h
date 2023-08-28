#ifndef _CONV2D_H
#define _CONV2D_H

#include "mat.h"
#include "helper.h"

typedef struct Conv2d {

    int n; // Number of samples in the minibatch
    int h_in;
    int w_in;
    int c_in;

    int kh;
    int kw;

    int h_out;
    int w_out;
    int c_out;

    int stride;

    Mat* inputs_d;
    Mat* inputs_lowered;
    Mat* inputs_lowered_d;
    Mat* conv_hnwc_z;
    Mat* conv_hnwc_z_prime;
    Mat* conv_hnwc_a;
    Mat* conv_hnwc_d;

    Mat* kernels;
    Mat* biases;

} Conv2d;

Conv2d* conv2d_initialize_normal(Conv2d* conv2d, float mean, float sd) {
    float* norm_args[2];
    norm_args[0] = &mean;
    norm_args[1] = &sd;
    mat_fill_func(conv2d->kernels, conv2d->kernels, mat_norm_filler_cb, norm_args);
    mat_fill(conv2d->biases, 0.0);
    return conv2d;
}

Conv2d* conv2d_malloc(int n, int h_in, int w_in, int c_in, int kh, int kw, int stride, int c_out) {

    if (!is_divisible(h_in - kh, stride)) {
        printf("Dimensions not compatible... h_in: %d, kh: %d, s: %d", h_in, kh, stride);
        exit(1);
    }

    if (!is_divisible(w_in - kw, stride)) {
        printf("Dimensions not compatible... w_in: %d, kw: %d, s: %d", w_in, kw, stride);
        exit(1);
    }

    Conv2d* conv2d  = (Conv2d*)malloc(sizeof(Conv2d));
    conv2d->n       = n;
    conv2d->h_in    = h_in;
    conv2d->w_in    = w_in;
    conv2d->c_in    = c_in;
    conv2d->kh      = kh;
    conv2d->kw      = kw;
    conv2d->stride  = stride;
    conv2d->c_out   = c_out;

    int h_out       = (h_in - kh) / stride + 1;
    conv2d->h_out   = h_out;
    int w_out       = (w_in - kw) / stride + 1;
    conv2d->w_out   = w_out;

    conv2d->inputs_d            = mat_malloc(n * h_in, w_in * c_in);
    conv2d->inputs_lowered      = mat_malloc(n * w_out, h_in * kw * c_in);
    conv2d->inputs_lowered_d    = mat_malloc(n * w_out, h_in * kw * c_in);
    conv2d->conv_hnwc_z         = mat_malloc(h_out, n * w_out * c_out);
    conv2d->conv_hnwc_z_prime   = mat_malloc(h_out, n * w_out * c_out);
    conv2d->conv_hnwc_a         = mat_malloc(h_out, n * w_out * c_out);
    conv2d->conv_hnwc_d         = mat_malloc(h_out, n * w_out * c_out);

    conv2d->kernels = mat_malloc(kh * kw * c_in, c_out);
    conv2d->biases  = mat_calloc(1, c_out);

    // mat_fill_func(conv2d->kernels, conv2d->kernels, mat_standard_norm_filler_cb, NULL);
    conv2d_initialize_normal(conv2d, 0, 0.01);

    return conv2d;
}

void conv2d_free(Conv2d* conv2d) {
    mat_free(conv2d->inputs_d);
    mat_free(conv2d->inputs_lowered);
    mat_free(conv2d->inputs_lowered_d);
    mat_free(conv2d->conv_hnwc_z);
    mat_free(conv2d->conv_hnwc_z_prime);
    mat_free(conv2d->conv_hnwc_a);
    mat_free(conv2d->conv_hnwc_d);
    mat_free(conv2d->kernels);
    mat_free(conv2d->biases);
    free(conv2d);
}

Conv2d* conv2d_set_n(Conv2d* conv2d, int n) {
    conv2d->n = n;

    mat_free(conv2d->inputs_d);
    mat_free(conv2d->inputs_lowered);
    mat_free(conv2d->inputs_lowered_d);
    mat_free(conv2d->conv_hnwc_z);
    mat_free(conv2d->conv_hnwc_z_prime);
    mat_free(conv2d->conv_hnwc_a);
    mat_free(conv2d->conv_hnwc_d);

    conv2d->inputs_d            = mat_malloc(n * conv2d->h_in, conv2d->w_in * conv2d->c_in);
    conv2d->inputs_lowered      = mat_malloc(n * conv2d->w_out, conv2d->h_in * conv2d->kw * conv2d->c_in);
    conv2d->inputs_lowered_d    = mat_malloc(n * conv2d->w_out, conv2d->h_in * conv2d->kw * conv2d->c_in);
    conv2d->conv_hnwc_z         = mat_malloc(conv2d->h_out, n * conv2d->w_out * conv2d->c_out);
    conv2d->conv_hnwc_z_prime   = mat_malloc(conv2d->h_out, n * conv2d->w_out * conv2d->c_out);
    conv2d->conv_hnwc_a         = mat_malloc(conv2d->h_out, n * conv2d->w_out * conv2d->c_out);
    conv2d->conv_hnwc_d         = mat_malloc(conv2d->h_out, n * conv2d->w_out * conv2d->c_out);
    
    return conv2d;
}

Conv2d* conv2d_lower(Conv2d* conv2d, Mat* inputs, Mat** inputs_sep) {

    int N = conv2d->n;
    int H = conv2d->h_in;
    int W = conv2d->w_in;
    int C = conv2d->c_in;
    int kw = conv2d->kw;
    int stride = conv2d->stride;

    int w_out = conv2d->w_out;

    Mat* lowered = conv2d->inputs_lowered;
    float* cur_lowered_data = lowered->data;

    float* inputs_data;
    if (inputs_sep == NULL)
        inputs_data = inputs->data;

    for (int n = 0; n < N; n++) {
        
        float* cur_input_start;
        if (inputs_sep == NULL)
            cur_input_start = inputs_data + n * H * W * C;
        else
            cur_input_start = inputs_sep[n]->data;

        for (int w_o = 0; w_o < w_out; w_o++) {
            int w_s = w_o * C * stride;
            for (int h_i = 0; h_i < H; h_i++) {
                memcpy(cur_lowered_data, cur_input_start + w_s + h_i * W * C, sizeof(float) * kw * C);
                cur_lowered_data += kw * C;
            }
        }
    }

    return conv2d;
}

// TODO: try implementing this with SIMD (AVX2) to see if it improves performance
Conv2d* conv2d_bias_forward(Conv2d* conv2d) {
    int c_out = conv2d->c_out;

    int pixels = conv2d->n * conv2d->h_out * conv2d->w_out;

    float* cur_result_data = conv2d->conv_hnwc_z->data;
    float* bias_data = conv2d->biases->data;

    for (int p = 0; p < pixels; p++) {
        for (int c = 0; c < c_out; c++) {
            *cur_result_data = bias_data[c];
            cur_result_data++;
        }
    }

    return conv2d;
}

Conv2d* conv2d_mm_forward(Conv2d* conv2d) {
    int n = conv2d->n;
    int c_in = conv2d->c_in;
    int h_out = conv2d->h_out;
    int w_out = conv2d->w_out;
    int c_out = conv2d->c_out;
    int stride = conv2d->stride;
    int kw = conv2d->kw;

    Mat* input = conv2d->inputs_lowered;
    Mat* result = conv2d->conv_hnwc_z;
    Mat* kernel = conv2d->kernels;

    for (int h = 0; h < h_out; h++) {
        float* input_start = input->data + h * stride * c_in * kw;
        float* result_start = result->data + h * w_out * c_out * n;

        cblas_sgemm(CblasRowMajor, 
                CblasNoTrans, CblasNoTrans,
                input->rows, kernel->cols, kernel->rows,
                1.0, input_start, input->cols, kernel->data, kernel->cols,
                1.0, result_start, kernel->cols);
    }

    return conv2d;
}

Conv2d* conv2d_forward(Conv2d* conv2d, Mat* inputs, Mat** inputs_sep) {
    conv2d_lower(conv2d, inputs, inputs_sep);
    conv2d_bias_forward(conv2d);
    conv2d_mm_forward(conv2d);
    mat_fill_func(conv2d->conv_hnwc_a, conv2d->conv_hnwc_z, nn_mat_relu_cb, NULL);
    return conv2d;
}

Conv2d* conv2d_backward_into_lowered(Conv2d* conv2d) {
    int c_out = conv2d->c_out;
    int c_in = conv2d->c_in;
    int N = conv2d->n;
    int w_out = conv2d->w_out;
    int stride = conv2d->stride;
    int kw = conv2d->kw;

    Mat* inputs_lowered_d = conv2d->inputs_lowered_d;
    Mat* kernels = conv2d->kernels;

    mat_fill(inputs_lowered_d, 0.0);

    for (int h = 0; h < conv2d->h_out; h++) {

        // float* other_start = other->mat->data + h * stride * C_in * kw;
        float* next_grad_start = conv2d->conv_hnwc_d->data + h * w_out * c_out * N;
        float* lowered_d_start = conv2d->inputs_lowered_d->data + h * stride * c_in * kw;

        cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                inputs_lowered_d->rows, kernels->rows, c_out,
                1.0, next_grad_start, c_out, kernels->data, c_out,
                1.0, lowered_d_start, inputs_lowered_d->cols);
    }

    return conv2d;
}

Mat* conv2d_backward_into_inputs(Conv2d* conv2d) {

    int N = conv2d->n;
    int H = conv2d->h_in;
    int W = conv2d->w_in;
    int C = conv2d->c_in;
    int kw = conv2d->kw;
    int stride = conv2d->stride;

    int out_w = conv2d->w_out;

    // Tensor* result = tensor_calloc(N, H, W, C);
    Mat* result = conv2d->inputs_d;

    float* mecced_start =  conv2d->inputs_lowered_d->data;

    mat_fill(result, 0.0);

    for (int n = 0; n < N; n++) {

        float* cur_start_res = result->data + n * H * W * C;

        for (int w_o = 0; w_o < out_w; w_o++) {
            int w_s = w_o * C * stride;
            for (int h_i = 0; h_i < H; h_i++) {

                for (int kw_i = 0; kw_i < kw * C; kw_i++) {
                    cur_start_res[w_s + h_i * W * C + kw_i] += *mecced_start;
                    mecced_start++;
                }
            }
        }
    }

    // mat_print(result);

    return result;
}

Conv2d* conv2d_update_weights_and_biases(Conv2d* conv2d, float lr) {
    int C_in = conv2d->c_in;
    int H_out = conv2d->h_out;
    int N_out = conv2d->n;
    int W_out = conv2d->w_out;
    int C_out = conv2d->c_out;
    int stride = conv2d->stride;
    int kw = conv2d->kw;
    
    int next_grad_rows = W_out * N_out;
    int next_grad_cols = C_out;

    int H = H_out;

    Mat* other = conv2d->inputs_lowered;
    Mat* next_grad = conv2d->conv_hnwc_d;

    Mat* kernels = conv2d->kernels;

    for (int h = 0; h < H; h++) {

        float* other_start = other->data + h * stride * C_in * kw;
        float* next_grad_start = next_grad->data + h * W_out * C_out * N_out;

        cblas_sgemm(CblasRowMajor,
                CblasTrans, CblasNoTrans,
                kernels->rows, kernels->cols, next_grad_rows,
                -lr / N_out, other_start, other->cols, next_grad_start, next_grad_cols,
                1.0, kernels->data, C_out);
    }

    /////////////////////////// BIASES

    Mat* biases_d = mat_calloc(conv2d->biases->rows, conv2d->biases->cols);
    float* biases_d_data = biases_d->data;

    int pixels = conv2d->n * conv2d->h_out * conv2d->w_out;

    float* cur_grad_data = conv2d->conv_hnwc_d->data;

    for (int p = 0; p < pixels; p++) {
        for (int c = 0; c < C_out; c++) {
            biases_d_data[c] += *cur_grad_data; 
            cur_grad_data++;
        }
    }

    biases_d = mat_scale(biases_d, biases_d, -lr / N_out);
    conv2d->biases = mat_add(conv2d->biases, conv2d->biases, biases_d);

    mat_free(biases_d);

    return conv2d;
}

#endif