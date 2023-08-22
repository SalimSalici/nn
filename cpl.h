#ifndef _CPL_H
#define _CPL_H

#include "mat.h"
#include "nn.h"

// ConvPoolLayer
typedef struct CPL {
    int group_count;
    int feature_count;
    int kernel_stride;
    int maxpool_stride;
    int kernel_side;
    int maxpool_side;

    Mat** inputs;
    int input_original_side;
    int input_rows;
    int input_cols;

    Mat** featuremaps_z;
    Mat** featuremaps_a;
    Mat** featuremaps_d;
    int featuremap_side;

    Mat** maxpoolmaps_a;
    Mat** maxpoolmaps_d;
    int maxpoolmap_side;

    Mat* kernels;
    int kernels_rows;
    int kernels_cols;

    float* biases;

    Mat* ones;
} CPL;

CPL* cpl_malloc(int group_count, int feature_count, int kernel_stride, int maxpool_stride,
    int input_original_side, int kernel_side, int maxpool_side) {

    CPL* cpl = (CPL*)malloc(sizeof(CPL));
    cpl->group_count = group_count;
    cpl->feature_count = feature_count;
    cpl->kernel_stride = kernel_stride;
    cpl->maxpool_stride = maxpool_stride;
    cpl->input_original_side = input_original_side;
    cpl->kernel_side = kernel_side;
    cpl->maxpool_side = maxpool_side;

    cpl->featuremap_side = (input_original_side - kernel_side) / kernel_stride + 1;
    cpl->maxpoolmap_side = (cpl->featuremap_side - maxpool_side) / maxpool_stride + 1;
    
    cpl->ones = mat_malloc(cpl->featuremap_side * cpl->featuremap_side, 1);
    mat_fill(cpl->ones, 1.0);

    cpl->inputs = (Mat**)malloc(sizeof(Mat*) * group_count);
    cpl->input_rows = kernel_side * kernel_side;
    cpl->input_cols = cpl->featuremap_side * cpl->featuremap_side;

    cpl->featuremaps_z = (Mat**)malloc(sizeof(Mat*) * group_count);
    cpl->featuremaps_a = (Mat**)malloc(sizeof(Mat*) * group_count);
    cpl->featuremaps_d = (Mat**)malloc(sizeof(Mat*) * group_count);
    cpl->maxpoolmaps_a = (Mat**)malloc(sizeof(Mat*) * group_count);
    cpl->maxpoolmaps_d = (Mat**)malloc(sizeof(Mat*) * group_count);

    cpl->kernels_rows = feature_count;
    cpl->kernels_cols = kernel_side * kernel_side;
    cpl->kernels = mat_malloc(cpl->kernels_rows, cpl->kernels_cols);
    mat_fill_func(cpl->kernels, cpl->kernels, mat_standard_norm_filler_cb, NULL);

    // cpl->biases = (float*)malloc(sizeof(float) * feature_count);
    cpl->biases = (float*)calloc(feature_count, sizeof(float));

    for (int i = 0; i < group_count; i++) {
        cpl->inputs[i] = mat_malloc(cpl->input_rows, cpl->input_cols);
        cpl->featuremaps_z[i] = mat_malloc(cpl->feature_count, cpl->featuremap_side * cpl->featuremap_side);
        cpl->featuremaps_a[i] = mat_malloc(cpl->feature_count, cpl->featuremap_side * cpl->featuremap_side);
        cpl->featuremaps_d[i] = mat_malloc(cpl->feature_count, cpl->featuremap_side * cpl->featuremap_side);
        cpl->maxpoolmaps_a[i] = mat_malloc(cpl->feature_count, cpl->maxpoolmap_side * cpl->maxpoolmap_side);
        cpl->maxpoolmaps_d[i] = mat_malloc(cpl->feature_count, cpl->maxpoolmap_side * cpl->maxpoolmap_side);
    }

    return cpl;
}

void cpl_free(CPL* cpl) {
    for (int i = 0; i < cpl->group_count; i++) {
        mat_free(cpl->inputs[i]);
        mat_free(cpl->featuremaps_z[i]);
        mat_free(cpl->featuremaps_a[i]);
        mat_free(cpl->featuremaps_d[i]);
        mat_free(cpl->maxpoolmaps_a[i]);
        mat_free(cpl->maxpoolmaps_d[i]);
    }
    mat_free(cpl->kernels);
    free(cpl->biases);
    free(cpl->inputs);
    free(cpl->featuremaps_z);
    free(cpl->featuremaps_a);
    free(cpl->featuremaps_d);
    free(cpl->maxpoolmaps_a);
    free(cpl->maxpoolmaps_d);
    free(cpl);
}

Mat* cpl_im2col(Mat* res, Mat* inputs, int kernel_side, int kernel_stride) {

    int inputs_cols = inputs->cols;

    assert(inputs->rows == inputs_cols);
    assert((inputs_cols - kernel_side) % kernel_stride == 0);

    float* inputs_data = inputs->data;

    int res_cols = res->cols;
    float* res_data = res->data;

    int rc = 0;
    for (int r = 0; r <= inputs_cols - kernel_side; r += kernel_stride) {
        for (int c = 0; c <= inputs_cols - kernel_side; c += kernel_stride) {
            int rr = 0;
            for (int kr = 0; kr < kernel_side; kr++) {
                for (int kc = 0; kc < kernel_side; kc++) {
                    res_data[rc + res_cols * rr] = inputs_data[c + kc + (r + kr) * inputs_cols];
                    rr++;
                }
            }
            rc++;
        }
    }
    return res;
}

CPL* cpl_forward_conv(CPL* cpl, Mat* inputs[], int inputs_count) {
    assert(inputs_count <= cpl->group_count);

    Mat* ones = cpl->ones;
    Mat* biases = mat_malloc_nodata(cpl->feature_count, 1);
    biases->data = cpl->biases;

    for (int i = 0; i < inputs_count; i++) {
        // cpl->inputs[i] = cpl_im2col(cpl->inputs[i], inputs[i], cpl->kernel_side, cpl->kernel_stride);
        cpl->inputs[i] = inputs[i];

        // float* cur = cpl->featuremaps_z[i]->data;
        // for (int r = 0; r < cpl->featuremaps_z[i]->rows; r++) {
        //     float b = cpl->biases[r];

        //     for (int c = 0; c < cpl->featuremaps_z[i]->cols; c++) {
        //         *cur = b;
        //         cur++;
        //     }
        // }

        // Initialize featuremaps with biases
        // This substitutes the above commented lines of code
        cblas_sgemm(CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            biases->rows, ones->rows, 1,
            1, biases->data, biases->cols, ones->data, ones->rows,
            0, cpl->featuremaps_z[i]->data, cpl->featuremaps_z[i]->cols);

        // Convolution
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    cpl->kernels_rows, cpl->input_cols, cpl->kernels_cols,
                    1.0, cpl->kernels->data, cpl->kernels_cols, cpl->inputs[i]->data, cpl->inputs[i]->cols,
                    1.0, cpl->featuremaps_z[i]->data, cpl->featuremaps_z[i]->cols);

        mat_fill_func(cpl->featuremaps_a[i], cpl->featuremaps_z[i], nn_mat_relu_cb, NULL);
    }

    free(biases);

    return cpl;
}

CPL* cpl_forward_maxpool(CPL* cpl, int inputs_count) {
    assert(inputs_count <= cpl->group_count);
    for (int i = 0; i < inputs_count; i++) {
        Mat* current_featuremap = cpl->featuremaps_a[i];
        Mat* current_maxpoolmap = cpl->maxpoolmaps_a[i];

        for (int f = 0; f < cpl->feature_count; f++) {
            Mat featuremap = {
                .rows = cpl->featuremap_side,
                .cols = cpl->featuremap_side,
                .data = current_featuremap->data + f * (cpl->featuremap_side * cpl->featuremap_side),
                .t = 0,
                .right = 1,
                .down = cpl->featuremap_side
            };

            Mat maxpoolmap = {
                .rows = cpl->maxpoolmap_side,
                .cols = cpl->maxpoolmap_side,
                .data = current_maxpoolmap->data + f * (cpl->maxpoolmap_side * cpl->maxpoolmap_side),
                .t = 0,
                .right = 1,
                .down = cpl->maxpoolmap_side
            };

            float* maxpoolmap_data = maxpoolmap.data;
            for (int r = 0; r < cpl->featuremap_side; r += cpl->maxpool_stride) {
                for (int c = 0; c < cpl->featuremap_side; c += cpl->maxpool_stride) {
                    float max = -INFINITY;
                    for (int rr = 0; rr < cpl->maxpool_side; rr++) {
                        for (int cc = 0; cc < cpl->maxpool_side; cc++) {
                            int idx = c + cc + (r + rr) * featuremap.down;
                            float cur = featuremap.data[idx];
                            if (cur > max) max = cur;
                        }
                    }
                    *maxpoolmap_data = max;
                    maxpoolmap_data++;
                }
            }
        }
    }
    return cpl;
}

CPL* cpl_forward(CPL* cpl, Mat* inputs[], int inputs_count) {
    cpl_forward_conv(cpl, inputs, inputs_count);
    cpl_forward_maxpool(cpl, inputs_count);
    return cpl;
}

CPL* cpl_backward(CPL* cpl, Mat* ds, int inputs_count) {
    
    assert(ds->cols = inputs_count);

    int ds_down = ds->down;

    for (int i = 0; i < inputs_count; i++) {
        float* cur_ds = ds->data + i;
        Mat* cur_fm_z = cpl->featuremaps_z[i];
        Mat* cur_fm_a = cpl->featuremaps_a[i];
        Mat* cur_fm_d = cpl->featuremaps_d[i];

        Mat* cur_mp_a = cpl->maxpoolmaps_a[i];

        for (int f = 0; f < cpl->feature_count; f++) {

            Mat f_featuremap_a = {
                .rows = cpl->featuremap_side,
                .cols = cpl->featuremap_side,
                .data = cur_fm_a->data + f * (cpl->featuremap_side * cpl->featuremap_side),
                .t = 0,
                .right = 1,
                .down = cpl->featuremap_side
            };

            Mat f_featuremap_d = {
                .rows = cpl->featuremap_side,
                .cols = cpl->featuremap_side,
                .data = cur_fm_d->data + f * (cpl->featuremap_side * cpl->featuremap_side),
                .t = 0,
                .right = 1,
                .down = cpl->featuremap_side
            };

            Mat f_maxpoolmap_a = {
                .rows = cpl->maxpoolmap_side,
                .cols = cpl->maxpoolmap_side,
                .data = cur_mp_a->data + f * (cpl->maxpoolmap_side * cpl->maxpoolmap_side),
                .t = 0,
                .right = 1,
                .down = cpl->maxpoolmap_side
            };

            float* f_fm_a_data = f_featuremap_a.data;
            float* f_fm_d_data = f_featuremap_d.data;
            float* f_mp_a_data = f_maxpoolmap_a.data;

            for (int r = 0; r < f_maxpoolmap_a.rows; r++) {
                for (int c = 0; c < f_maxpoolmap_a.cols; c++) {
                    if (f_fm_a_data[c*2 + r*2 * cpl->featuremap_side] == f_mp_a_data[c + r * f_maxpoolmap_a.cols]) {
                        
                        f_fm_d_data[c*2 + r*2 * cpl->featuremap_side] = *(cur_ds + ds_down * (c + r * f_maxpoolmap_a.cols));
                        f_fm_d_data[c*2+1 + (r*2+1) * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2 + (r*2+1) * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2+1 + r*2 * cpl->featuremap_side] = 0;

                    } else if (f_fm_a_data[c*2+1 + (r*2+1) * cpl->featuremap_side] == f_mp_a_data[c + r * f_maxpoolmap_a.cols]) {

                        f_fm_d_data[c*2 + r*2 * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2+1 + (r*2+1) * cpl->featuremap_side] = *(cur_ds + ds_down * (c + r * f_maxpoolmap_a.cols));
                        f_fm_d_data[c*2 + (r*2+1) * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2+1 + r*2 * cpl->featuremap_side] = 0;

                    } else if (f_fm_a_data[c*2 + (r*2+1) * cpl->featuremap_side] == f_mp_a_data[c + r * f_maxpoolmap_a.cols]) {

                        f_fm_d_data[c*2 + r*2 * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2+1 + (r*2+1) * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2 + (r*2+1) * cpl->featuremap_side] = *(cur_ds + ds_down * (c + r * f_maxpoolmap_a.cols));
                        f_fm_d_data[c*2+1 + r*2 * cpl->featuremap_side] = 0;

                    } else if (f_fm_a_data[c*2+1 + r*2 * cpl->featuremap_side] == f_mp_a_data[c + r * f_maxpoolmap_a.cols]) {

                        f_fm_d_data[c*2 + r*2 * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2+1 + (r*2+1) * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2 + (r*2+1) * cpl->featuremap_side] = 0;
                        f_fm_d_data[c*2+1 + r*2 * cpl->featuremap_side] = *(cur_ds + ds_down * (c + r * f_maxpoolmap_a.cols));

                    }
                }
            }

            cur_ds = cur_ds + ds_down * f_maxpoolmap_a.rows * f_maxpoolmap_a.cols;
        }

        Mat* cur_fm_z_prime = mat_fill_func(NULL, cur_fm_z, nn_mat_relu_prime_cb, NULL);
        mat_hadamard_prod(cur_fm_d, cur_fm_d, cur_fm_z_prime, 1.0);
        mat_free(cur_fm_z_prime);
    }
    return cpl;
}

CPL* cpl_update_weights_and_biases(CPL* cpl, float lr, float lambda, int training_count, int inputs_count) {
    
    Mat* kernels_gradient = mat_malloc(cpl->kernels->cols, cpl->kernels->rows);
    
    Mat* cur_inputs = cpl->inputs[0];
    Mat* cur_gradients = mat_transpose(cpl->featuremaps_d[0]);

    mat_mult_mm(kernels_gradient, cur_inputs, cur_gradients, 1.0, 0.0);

    for (int i = 1; i < inputs_count; i++) {
        Mat* cur_inputs = cpl->inputs[i];
        Mat* cur_gradients = mat_transpose(cpl->featuremaps_d[i]);

        mat_mult_mm(kernels_gradient, cur_inputs, cur_gradients, 1.0, 1.0);
    }

    Mat* biases_gradient = mat_malloc(cpl->featuremaps_d[0]->rows, 1);

    mat_transpose(cpl->featuremaps_d[0]);
    mat_mult_mv(biases_gradient, cpl->featuremaps_d[0], cpl->ones, 1.0, 0.0);

    for (int i = 1; i < inputs_count; i++) {
        mat_transpose(cpl->featuremaps_d[i]);
        mat_mult_mv(biases_gradient, cpl->featuremaps_d[i], cpl->ones, 1.0, 1.0);
    }

    for (int f = 0; f < cpl->feature_count; f++) {
        cpl->biases[f] -= (lr / inputs_count) * biases_gradient->data[f];
    }

    mat_scale(kernels_gradient, kernels_gradient, lr / inputs_count);
    mat_sub(cpl->kernels, cpl->kernels, mat_transpose(kernels_gradient));
    mat_free(kernels_gradient);
    mat_free(biases_gradient);
    // mat_free(ones);

    return cpl;
}

#endif