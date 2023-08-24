#ifndef _TENSOR_H
#define _TENSOR_H

#include "mat.h"

#define tensor_view_mat(tens, rows, cols) mat_view((tens)->mat, (rows), (cols))

typedef struct Tensor4 {
    int dim0;
    int dim1;
    int dim2;
    int dim3;
    float* data;
    Mat* mat;
} Tensor;

Tensor* tensor_malloc(int dim0, int dim1, int dim2, int dim3) {
    Tensor* tens = (Tensor*)malloc(sizeof(Tensor));
    tens->dim0 = dim0;
    tens->dim1 = dim1;
    tens->dim2 = dim2;
    tens->dim3 = dim3;
    tens->data = (float*)malloc(sizeof(float) * dim0 * dim1 * dim2 * dim3);
    tens->mat = mat_malloc_nodata(dim0 * dim1, dim2 * dim3);
    tens->mat->data = tens->data;
    return tens;
}

Tensor* tensor_calloc(int dim0, int dim1, int dim2, int dim3) {
    Tensor* tens = (Tensor*)malloc(sizeof(Tensor));
    tens->dim0 = dim0;
    tens->dim1 = dim1;
    tens->dim2 = dim2;
    tens->dim3 = dim3;
    tens->data = (float*)calloc(dim0 * dim1 * dim2 * dim3, sizeof(float));
    tens->mat = mat_malloc_nodata(dim0 * dim1, dim2 * dim3);
    tens->mat->data = tens->data;
    return tens;
}

Tensor* tensor_malloc_nodata(int dim0, int dim1, int dim2, int dim3) {
    Tensor* tens = (Tensor*)malloc(sizeof(Tensor));
    tens->dim0 = dim0;
    tens->dim1 = dim1;
    tens->dim2 = dim2;
    tens->dim3 = dim3;
    tens->data = NULL;
    tens->mat = mat_malloc_nodata(dim0 * dim1, dim2 * dim3);
    tens->mat->data = tens->data;
    return tens;
}

void tensor_free(Tensor* t) {
    // this frees up tens->data too
    mat_free(t->mat);

    free(t);
}

Tensor* tensor_attach_data(Tensor* tens, float* data) {
    tens->data = data;
    tens->mat->data = data;
    return tens;
}

// Tensor* tensor_view_mat(Tensor* tens, int rows, int cols) {
//     mat_view(tens->mat, rows, cols);
//     return tens;
// }

#endif