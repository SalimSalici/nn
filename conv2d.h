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

    Mat* inputs;
    Mat* inputs_lowered;
    Mat* inputs_lowered_d;
    Mat* conv_hnwc_z;
    Mat* conv_hnwc_a;
    Mat* conv_hnwc_d;

    Mat* kernels;
    Mat* biases;

} Conv2d;

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

    int h_out = (h_in - kh) / stride + 1;
    conv2d->h_out = h_out;
    int w_out = (w_in - kw) / stride + 1;
    conv2d->w_out = w_out;

    conv2d->inputs_lowered = mat_malloc(n * w_out, h_in * kw * c_in);
    conv2d->inputs_lowered_d = mat_malloc(n * w_out, h_in * kw * c_in);
    conv2d->conv_hnwc_z = mat_malloc(h_out, n * w_out * c_out);
    conv2d->conv_hnwc_a = mat_malloc(h_out, n * w_out * c_out);
    conv2d->conv_hnwc_d = mat_malloc(h_out, n * w_out * c_out);

    conv2d->kernels = mat_malloc(kh * kw * c_in, c_out);
    conv2d->biases = mat_calloc(1, c_out);

    mat_fill_func(conv2d->kernels, conv2d->kernels, mat_standard_norm_filler_cb, NULL);

    return conv2d;
}

void conv2d_free(Conv2d* conv2d) {
    mat_free(conv2d->inputs_lowered);
    mat_free(conv2d->inputs_lowered_d);
    mat_free(conv2d->conv_hnwc_z);
    mat_free(conv2d->conv_hnwc_a);
    mat_free(conv2d->conv_hnwc_d);
    mat_free(conv2d->kernels);
    mat_free(conv2d->biases);
    free(conv2d);
}

Conv2d* conv2d_lower(Conv2d* conv2d, Mat* inputs) {

    int N = conv2d->n;
    int H = conv2d->h_in;
    int W = conv2d->w_in;
    int C = conv2d->c_in;
    int kw = conv2d->kw;
    int stride = conv2d->stride;

    int w_out = conv2d->w_out;

    Mat* lowered = conv2d->inputs_lowered;
    float* cur_lowered_data = lowered->data;

    float* inputs_data = inputs->data;

    for (int n = 0; n < N; n++) {
        float* cur_input_start = inputs_data + n * H * W * C;
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

Conv2d* conv2d_forward(Conv2d* conv2d, Mat* inputs) {
    conv2d_lower(conv2d, inputs);
    conv2d_bias_forward(conv2d);
    conv2d_mm_forward(conv2d);
    mat_fill_func(conv2d->conv_hnwc_a, conv2d->conv_hnwc_z, nn_mat_relu_cb, NULL);
    return conv2d;
}

#endif