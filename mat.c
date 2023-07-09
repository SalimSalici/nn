#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "helper.h"

#define MAT_ELEM_IDX(m, row, col) row * m->cols + col

float mat_standard_norm_filler_cb(float cur, int row, int col, void* func_args) {
    return gauss();
}

float mat_zero_filler_cb(float cur, int row, int col, void* func_args) {
    return 0.0;
}

// func_args is an array of two floats
float mat_norm_filler_cb(float cur, int row, int col, void* func_args) {
    assert(func_args != NULL);
    float** args;
    args = (float**)func_args;
    float mean = *(args[0]);
    float stand_dev = *(args[1]);
    return gauss() * stand_dev + mean;
}

void mat_print(Mat* m) {
    float* cur = m->data;
    for (int r = 0; r < m->rows; r++) {
        printf("{");
        for (int c = 0; c < m->cols; c++) {
            printf("%f, ", *cur);
            cur++;
        }
        printf("}\n");
    }
}

void mat_print_row(Mat* m, int row) {
    float* cur = m->data + row * m->cols;
    for (int r = 0; r < m->cols; r++) {
        printf("%f\t", *cur);
        cur++;
    }
}

void mat_print_col(Mat* m, int col) {
    float* cur = m->data + col;
    for (int r = 0; r < m->rows; r++) {
        printf("%f\t", *cur);
        cur += m->cols;
        printf("\n");
    }
}

Mat* mat_malloc(int rows, int cols) {
    Mat* m = (Mat*)malloc(sizeof(Mat));
    m->rows =rows;
    m->cols = cols;
    m->data = (float*)malloc(sizeof(float) *rows * cols);
    return m;
}

Mat* mat_malloc_cpy(int rows, int cols, float cpy[rows][cols]) {
    Mat* m = (Mat*)malloc(sizeof(Mat));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(sizeof(float) * rows * cols);
    memcpy(m->data, cpy, sizeof(float) * rows * cols);
    return m;
}

Mat* mat_cpy(Mat* m) {
    Mat* cpy = mat_malloc(m->rows, m->cols);
    memcpy(cpy->data, m->data, sizeof(float) * m->rows * m->cols);
    return cpy;
}

void mat_free(Mat* m) {
    free(m->data);
    free(m);
    return;
}

Mat* mat_mult(Mat* res, Mat* a, Mat* b) {
    assert(a->cols == b->rows);
    if (res == NULL) {
        res = mat_malloc(a->rows, b->cols);
    } else {
        assert(res->rows == a->rows);
        assert(res->cols == b->cols);
    }
    float* cur = res->data;
    for (int r = 0; r < res->rows; r++) {
        for (int c = 0; c < res->cols; c++) {
            float sum = 0;
            float *a_cur = a->data + r * a->cols;
            float *b_cur = b->data + c;
            for (int k = 0; k < a->cols; k++) {
                sum += (*a_cur) * (*b_cur);
                a_cur++;
                b_cur += b->cols; 
            }
            *cur = sum;
            cur++;
        }
    }
    return res;
}

Mat* mat_add(Mat* res, Mat* a, Mat* b) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    if (res == NULL) {
        res = mat_malloc(a->rows, a->cols);
    } else {
        assert(res->rows == b->rows);
        assert(res->cols == b->cols);
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        *(res->data + i) = *(a->data + i) + *(b->data + i);
    }
    return res;
}

Mat* mat_sub(Mat* res, Mat* a, Mat* b) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    if (res == NULL) {
        res = mat_malloc(a->rows, a->cols);
    } else {
        assert(res->rows == b->rows);
        assert(res->cols == b->cols);
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        *(res->data + i) = *(a->data + i) - *(b->data + i);
    }
    return res;
}

Mat* mat_hadamard_prod(Mat* res, Mat* a, Mat* b) {
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    if (res == NULL) {
        res = mat_malloc(a->rows, a->cols);
    } else {
        assert(res->rows == b->rows);
        assert(res->cols == b->cols);
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        *(res->data + i) = (*(a->data + i)) * (*(b->data + i));
    }
    return res;
}

Mat* mat_transpose(Mat* res, Mat* a) {
    if (res == NULL) {
        res = mat_malloc(a->cols, a->rows);
    } else {
        assert(res->rows == a->cols);
        assert(res->cols == a->rows);
    }
    for (int row = 0; row < res->rows; row++)
        for (int col = 0; col < res->cols; col++)
            res->data[MAT_ELEM_IDX(res, row, col)] = a->data[(MAT_ELEM_IDX(a, col, row))];
    return res;
}

Mat* mat_scale(Mat* res, Mat* a, float f) {
    if (res == NULL) {
        res = mat_malloc(a->rows, a->cols);
    } else {
        assert(a->rows == res->rows);
        assert(a->cols == res->cols);
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        *(res->data + i) = *(a->data + i) * f;
    }
    return res;
}

Mat* mat_fill(Mat* m, float f) {
    float* cur = m->data;
    for (int r = 0; r < m->rows; r++) {
        for (int c = 0; c < m->cols; c++) {
            *cur = f;
            cur++;
        }
    }
    return m;
}

Mat* mat_fill_func(Mat* res, Mat* m, float (*f)(float, int, int, void*), void* func_args) {
    if (res == NULL) {
        res = mat_malloc(m->rows, m->cols);
    } else {
        assert(res->rows == m->rows);
        assert(res->cols == m->cols);
    }
    float* cur_res = res->data;
    float* cur = m->data;
    for (int r = 0; r < m->rows; r++) {
        for (int c = 0; c < m->cols; c++) {
            *cur_res = (*f)(*cur, r, c, func_args);
            cur_res++;
            cur++;
        }
    }
    return res;
}

Mat* mat_diag_fill(Mat* m, float f) {
    float* cur = m->data;
    for (int c = 0; c < m->cols; c++) {
        *cur = f;
        cur += m->cols + 1;
    }
    return m;
}

float mat_max(Mat* m) {
    float max = m->data[0];
    for (int i = 1; i < m->rows * m->cols; i++)
        if (m->data[i] > max) max = m->data[i];
    return max;
}