#ifndef _MEC_H
#define _MEC_H

// https://arxiv.org/pdf/1706.06873.pdf

#include <math.h>
#include "tensor.h"
#include "mat.h"

int mec_is_divisible(int x, int y) {
    return fabs(fmod((float)x, (float)y)) < 0.000001;
}

/*
*   Expects input to be a 4d tensor with dimensions (N, H, W, C) where:
*       N: Number of samples in the batch
*       H: Height of one sample
*       W: Width of one sample
*       C: Channels of one sample
*
*   Returns mec lowered matrix
*
*/
Tensor* im2mec_input(Tensor* input, int kh, int kw, int stride) {

    float* data = input->data;

    int N = input->dim0;
    int H = input->dim1;
    int W = input->dim2;
    int C = input->dim3;

    if (!mec_is_divisible(H - kh, stride)) {
        printf("Dimensions not compatible... H: %d, kh: %d, s: %d", H, kh, stride);
        exit(1);
    }
    
    if (!mec_is_divisible(W - kw, stride)) {
        printf("Dimensions not compatible... W: %d, kw: %d, s: %d", H, kh, stride);
        exit(1);
    }

    // int out_h = (H - kh) / stride + 1;
    int out_w = (W - kw) / stride + 1;

    Tensor* result = tensor_malloc(N, out_w, H * kw, C);
    float* cur_res = result->data;

    for (int n = 0; n < N; n++) {

        int start_n = n * H * W * C;
        float* cur_data_start = data + start_n;

        for (int w_o = 0; w_o < out_w; w_o++) {
            int w_s = w_o * C * stride;
            for (int h_i = 0; h_i < H; h_i++) {
                memcpy(cur_res, cur_data_start + w_s + h_i * W * C, sizeof(float) * kw * C);
                cur_res += kw * C;
            }
        }
    }
    return result;
}

/*
*   Input: matrix obatained from im2mec_input function
*   kernel: Tensor with dimensions (1, KH, KW, KC_out) where:
*       KH: Kernel height (rows)
*       KW: Kernel width (cols)
*       KC: Kernel output channels
*
*   Returns mec lowered matrix
*
*/
Tensor* mec_conv(Tensor* input, Mat* kernel, int kw, int stride) {
    int C_in = input->dim3;
    int H_out = input->dim1;
    int N_out = input->dim0;
    int W_out = H_out;
    int C_out = kernel->cols;

    Tensor* result = tensor_malloc(H_out, N_out, W_out, C_out);

    tensor_view_mat(input, input->dim0 * input->dim1, input->dim2 * input->dim3);

    for (int h = 0; h < H_out; h++) {
        float* input_start = input->mat->data + h * stride * C_in * kw;
        float* result_start = result->data + h * W_out * C_out * N_out;

        cblas_sgemm(CblasRowMajor, 
                CblasNoTrans, CblasNoTrans,
                input->mat->rows, kernel->cols, kernel->rows,
                1.0, input_start, input->mat->cols, kernel->data, kernel->cols,
                0.0, result_start, kernel->cols);
    }

    return result;
}

Mat* mec_conv_backwards_kernel(Tensor* other, Tensor* next_grad, int kh, int kw, int in_c, int stride) {

    int C_in = other->dim3;
    int H_out = other->dim1;
    int N_out = other->dim0;
    int W_out = H_out;
    int C_out = next_grad->dim3;

    Mat* grad = mat_calloc(kh * kw * in_c, C_out);

    tensor_view_mat(other, other->dim0 * other->dim1, other->dim2 * other->dim3);
    
    int next_grad_rows = next_grad->dim1 * next_grad->dim2;
    int next_grad_cols = C_out;

    int H = next_grad->dim0;

    for (int h = 0; h < H; h++) {

        float* other_start = other->mat->data + h * stride * C_in * kw;
        float* next_grad_start = next_grad->data + h * W_out * C_out * N_out;

        cblas_sgemm(CblasRowMajor, 
                CblasTrans, CblasNoTrans,
                grad->rows, grad->cols, next_grad_rows,
                1.0, other_start, other->mat->cols, next_grad_start, next_grad_cols,
                1.0, grad->data, C_out);
    }

    return grad;
}

Tensor* mec_conv_backwards_prev(Mat* kernel, Tensor* next_grad, int kh, int kw, int in_c, int stride, int in_h) {

    // int C_in = other->dim3;
    // int H_out = other->dim1;
    // int N_out = other->dim0;
    // int W_out = H_out;
    int C_out = next_grad->dim3;
    int N = next_grad->dim1;
    int W_out = next_grad->dim2;

    Tensor* grad = tensor_calloc(N, W_out, kw * in_c, in_h);

    tensor_view_mat(grad, N * W_out, kw * in_c * in_h);
    // mat_print(grad->mat);
    // exit(0);
    
    // int next_grad_rows = next_grad->dim1 * next_grad->dim2;
    // int next_grad_cols = C_out;

    // int H = next_grad->dim0;

    for (int h = 0; h < next_grad->dim0; h++) {

        // float* other_start = other->mat->data + h * stride * C_in * kw;
        float* next_grad_start = next_grad->data + h * W_out * C_out * N;
        float* grad_start = grad->data + h * stride * in_c * kw;

        cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                grad->mat->rows, kernel->rows, C_out,
                1.0, next_grad_start, C_out, kernel->data, C_out,
                1.0, grad_start, grad->mat->cols);
    }

    return grad;
}

Tensor* mec2im_input_additive(Tensor* mecced, int kh, int kw, int stride) {

    int N = mecced->dim0;
    int H = mecced->dim2 / kw;
    int W = H;
    int C = mecced->dim3;

    int out_w = (W - kw) / stride + 1;

    Tensor* result = tensor_calloc(N, H, W, C);

    float* mecced_start = mecced->data;

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
    return result;
}

#endif