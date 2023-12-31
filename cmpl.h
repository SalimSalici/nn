#ifndef _CMPL_H
#define _CMPL_H

#include "conv2d.h"
#include "maxpool.h"

typedef struct Cmpl {
    Conv2d* conv2d;
    Maxpool* maxpool;
    int is_first;
} Cmpl;

Cmpl* cmpl_malloc(int h_in, int w_in, int c_in, int kh, int kw, int kstride, int c_out,
                  int mh, int mw, int mstride) {

    if (!is_divisible(h_in - kh, kstride)) {
        printf("cmpl_malloc: Dimensions not compatible... h_in: %d, kh: %d, s: %d", h_in, kh, kstride);
        exit(1);
    }

    if (!is_divisible(w_in - kw, kstride)) {
        printf("cmpl_malloc: Dimensions not compatible... w_in: %d, kw: %d, s: %d", w_in, kw, kstride);
        exit(1);
    }

    int conv_h_out = (h_in - kh) / kstride + 1;
    int conv_w_out = (w_in - kw) / kstride + 1;

    if (!is_divisible(conv_h_out - mh, mstride)) {
        printf("cmpl_malloc: Dimensions not compatible between conv2d and maxpool...\n");
        exit(1);
    }

    if (!is_divisible(conv_w_out - mw, mstride)) {
        printf("cmpl_malloc: Dimensions not compatible between conv2d and maxpool...\n");
        exit(1);
    }

    Cmpl* cmpl = (Cmpl*)malloc(sizeof(Cmpl));
    cmpl->conv2d = conv2d_malloc(h_in, w_in, c_in, kh, kw, kstride, c_out);
    cmpl->maxpool = maxpool_malloc(conv_h_out, conv_w_out, c_out, mh, mw, mstride);
    return cmpl;
}

Cmpl* cmpl_malloc_raw(Conv2d* conv2d, Maxpool* maxpool) {
    Cmpl* cmpl = (Cmpl*)malloc(sizeof(Cmpl));
    cmpl->conv2d = conv2d;
    cmpl->maxpool = maxpool;
    return cmpl;
}

void cmpl_free(Cmpl* cmpl) {
    conv2d_free(cmpl->conv2d);
    maxpool_free(cmpl->maxpool);
    free(cmpl);
}

Mat* cmpl_forward(Cmpl* cmpl, Mat* inputs, Mat** inputs_sep) {
    conv2d_forward(cmpl->conv2d, inputs, inputs_sep);
    maxpool_forward(cmpl->maxpool, cmpl->conv2d->conv_hnwc_a);
    return cmpl->maxpool->outputs;
}

Mat* cmpl_backward(Cmpl* cmpl, Mat* grad) {

    Conv2d* conv2d = cmpl->conv2d;

    maxpool_backward(cmpl->maxpool, conv2d, grad);

    conv2d->conv_hnwc_z_prime = mat_fill_func(conv2d->conv_hnwc_z_prime, conv2d->conv_hnwc_z, nn_mat_relu_prime_cb, NULL);
    conv2d->conv_hnwc_d = mat_hadamard_prod(conv2d->conv_hnwc_d, conv2d->conv_hnwc_d, conv2d->conv_hnwc_z_prime, 1.0);
    
    if (!cmpl->is_first) {
        conv2d_backward_into_lowered(conv2d);
        conv2d_backward_into_inputs(conv2d);
    }
    return conv2d->inputs_d;
}

Cmpl* cmpl_update_weights_and_biases(Cmpl* cmpl, float lr) {
    conv2d_update_weights(cmpl->conv2d, lr);
    conv2d_update_biases(cmpl->conv2d, lr);
    return cmpl;
}

#endif
