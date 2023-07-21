#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include <math.h>
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"

float mat_standard_norm_filler_cb(float cur, int row, int col, void* func_args) {
    return standard_gauss();
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
    return standard_gauss() * stand_dev + mean;
}

void mat_print(Mat* m) {
    int loop_rows = m->t ? m->cols : m->rows;
    int loop_cols = m->t ? m->rows : m->cols;
    for (int r = 0; r < loop_rows; r++) {
        float* cur = m->data + m->down * r;
        printf("{");
        for (int c = 0; c < loop_cols; c++) {
            if(*cur == 0)
                printf("z\t");
            else
                printf("%.2f\t", *cur);
            cur += m->right;
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
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(sizeof(float) *rows * cols);
    m->t = 0;
    m->right = 1;
    m->down = cols;
    return m;
}

Mat* mat_malloc_cpy(int rows, int cols, float cpy[rows][cols]) {
    Mat* m = (Mat*)malloc(sizeof(Mat));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(sizeof(float) * rows * cols);
    m->t = 0;
    m->right = 1;
    m->down = cols;
    memcpy(m->data, cpy, sizeof(float) * rows * cols);
    return m;
}

Mat* mat_cpy(Mat* m) {
    Mat* cpy = mat_malloc(m->rows, m->cols);
    memcpy(cpy->data, m->data, sizeof(float) * m->rows * m->cols);
    cpy->t = m->t;   
    cpy->right = m->right;
    cpy->down = m->down; 
    return cpy;
}

void mat_free(Mat* m) {
    free(m->data);
    free(m);
    return;
}

int mat_equals(Mat* a, Mat* b) {
    assert(!a->t && !b->t);
    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        if (a->data[i] != b->data[i]) return 0;
    }
    return 1;
}

Mat* mat_transpose(Mat* m) {
    int tmp = m->right;
    m->right = m->down;
    m->down = tmp;
    m->t = !(m->t);
    return m;
}

Mat* mat_mult_old(Mat* res, Mat* a, Mat* b) {
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

Mat* mat_mult(Mat* res, Mat* a, Mat* b) {

    if (a->t == b->t)
        assert(a->cols == b->rows);
    else {
        if (a->t) 
            assert(a->rows == b->rows);
        else
            assert(a->cols == b->cols);
    }
    if (res == NULL) {
        int rows, cols;

        if (a->t) rows = a->cols;
        else rows = a->rows;

        if (b->t) cols = b->rows;
        else cols = b->cols;

        res = mat_malloc(rows, cols);
    } else {
        if (a->t) assert(res->rows == a->cols);
        else assert(res->rows == a->rows);

        if (b->t) assert(res->cols == b->rows);
        else assert(res->cols == b->cols);

        assert(!res->t);
    }

    int loop_cols = a->t ? a->rows : a->cols;
    for (int r = 0; r < res->rows; r++) {
        for (int c = 0; c < res->cols; c++) {
            float sum = 0;
            float *cur_res = res->data + r * res->down + c * res->right;
            float *cur_a = a->data + r * a->down;
            float *cur_b = b->data + c * b->right;
            for (int k = 0; k < loop_cols; k++) {
                sum += (*cur_a) * (*cur_b);
                cur_a += a->right;
                cur_b += b->down; 
            }
            *cur_res = sum;
        }
    }
    return res;
}

Mat* mat_mult_mv(Mat* res, Mat* a, Mat* b, float ab_s, float c_s) {

    #ifdef MAT_USE_OPENBLAS

    assert(b->cols == 1);

    if (res == NULL) {
        int rows, cols;

        if (a->t) rows = a->cols;
        else rows = a->rows;

        cols = 1;

        res = mat_malloc(rows, cols);
    } else {
        if (a->t) assert(res->rows == a->cols);
        else assert(res->rows == a->rows);

        assert(res->cols == 1);
    }

    CBLAS_TRANSPOSE a_transposed = a->t ? CblasTrans : CblasNoTrans;

    cblas_sgemv(CblasRowMajor, 
                a_transposed,
                a->rows, a->cols,
                ab_s, a->data, a->cols, b->data, 1, 
                c_s, res->data, 1);

    return res;

    #else

    return mat_mult(res, a, b);

    #endif
}

Mat* mat_mult_mm(Mat* res, Mat* a, Mat* b, float ab_s, float c_s) {

    #ifdef MAT_USE_OPENBLAS

    if (res == NULL) {
        int rows, cols;

        if (a->t) rows = a->cols;
        else rows = a->rows;

        if (b->t) cols = b->rows;
        else cols = b->cols;

        res = mat_malloc(rows, cols);
    }

    CBLAS_TRANSPOSE a_transposed = a->t ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE b_transposed = b->t ? CblasTrans : CblasNoTrans;

    int a_rows = a->t ? a->cols : a->rows;
    int a_cols = a->t ? a->rows : a->cols;
    int b_cols = b->t ? b->rows : b->cols;

    cblas_sgemm(CblasRowMajor, 
                a_transposed, b_transposed,
                a_rows, b_cols, a_cols,
                ab_s, a->data, a->cols, b->data, b->cols,
                c_s, res->data, res->cols);

    return res;

    #else

    return mat_mult(res, a, b);

    #endif
}

Mat* mat_add_old(Mat* res, Mat* a, Mat* b) {
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

typedef struct MatAddThreadArgs {
    Mat* res;
    Mat* a;
    Mat* b;
    int row_start;
    int col_start;
    int row_end;
    int col_end;
} MatAddThreadArgs;

void* mat_add_thread(void* _args) {
    MatAddThreadArgs* args = (MatAddThreadArgs*)_args;
    Mat* res = args->res;
    Mat* a = args->a;
    Mat* b = args->b;
    for (int r = args->row_start; r < args->row_end; r++) {
        float* cur_res = res->data + res->down * r;
        float* cur_a = a->data + a->down * r;
        float* cur_b = b->data + b->down * r;
        for (int c = args->col_start; c < args->col_end; c++) {
            *cur_res = *cur_a + *cur_b;
            cur_res += res->right;
            cur_a += a->right;
            cur_b += b->right;
        }
    }
    return NULL;
}

Mat* mat_add_tr_supp(Mat* res, Mat* a, Mat* b) {
    if (a->t == b->t) {
        assert(a->rows == b->rows);
        assert(a->cols == b->cols);
    } else {
        assert(a->rows == b->cols);
        assert(a->cols == b->rows);
    }
    
    if (res == NULL) {
        if (!a->t)
            res = mat_malloc(a->rows, a->cols);
        else
            res = mat_malloc(a->cols, a->rows);
    } else {
        if (a->t == res->t) {
            assert(res->rows == a->rows);
            assert(res->cols == a->cols);
        } else {
            assert(res->rows == a->cols);
            assert(res->cols == a->rows);
        }
    }

    int loop_rows = res->t ? res->cols : res->rows;
    int loop_cols = res->t ? res->rows : res->cols;

    for (int r = 0; r < loop_rows; r++) {
        float* cur_res = res->data + res->down * r;
        float* cur_a = a->data + a->down * r;
        float* cur_b = b->data + b->down * r;
        for (int c = 0; c < loop_cols; c++) {
            *cur_res = *cur_a + *cur_b;
            cur_res += res->right;
            cur_a += a->right;
            cur_b += b->right;
        }
    }

    return res;
}

Mat* mat_add(Mat* res, Mat* a, Mat* b) {
    if (a->t || b->t) return mat_add_tr_supp(res, a, b);

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

Mat* mat_sub_tr_supp(Mat* res, Mat* a, Mat* b) {
    if (a->t == b->t) {
        assert(a->rows == b->rows);
        assert(a->cols == b->cols);
    } else {
        assert(a->rows == b->cols);
        assert(a->cols == b->rows);
    }
    
    if (res == NULL) {
        if (!a->t)
            res = mat_malloc(a->rows, a->cols);
        else
            res = mat_malloc(a->cols, a->rows);
    } else {
        if (a->t == res->t) {
            assert(res->rows == a->rows);
            assert(res->cols == a->cols);
        } else {
            assert(res->rows == a->cols);
            assert(res->cols == a->rows);
        }
    }

    int loop_rows = res->t ? res->cols : res->rows;
    int loop_cols = res->t ? res->rows : res->cols;

    for (int r = 0; r < loop_rows; r++) {
        float* cur_res = res->data + res->down * r;
        float* cur_a = a->data + a->down * r;
        float* cur_b = b->data + b->down * r;
        for (int c = 0; c < loop_cols; c++) {
            *cur_res = *cur_a - *cur_b;
            cur_res += res->right;
            cur_a += a->right;
            cur_b += b->right;
        }
    }

    return res;
}

Mat* mat_sub(Mat* res, Mat* a, Mat* b) {
    if (a->t || b->t) return mat_sub_tr_supp(res, a, b);

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

Mat* mat_sub_old(Mat* res, Mat* a, Mat* b) {
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

Mat* mat_hadamard_prod_tr_supp(Mat* res, Mat* a, Mat* b, float s) {
    if (a->t == b->t) {
        assert(a->rows == b->rows);
        assert(a->cols == b->cols);
    } else {
        assert(a->rows == b->cols);
        assert(a->cols == b->rows);
    }
    
    if (res == NULL) {
        if (!a->t)
            res = mat_malloc(a->rows, a->cols);
        else
            res = mat_malloc(a->cols, a->rows);
    } else {
        if (a->t == res->t) {
            assert(res->rows == a->rows);
            assert(res->cols == a->cols);
        } else {
            assert(res->rows == a->cols);
            assert(res->cols == a->rows);
        }
    }

    int loop_rows = res->t ? res->cols : res->rows;
    int loop_cols = res->t ? res->rows : res->cols;

    for (int r = 0; r < loop_rows; r++) {
        float* cur_res = res->data + res->down * r;
        float* cur_a = a->data + a->down * r;
        float* cur_b = b->data + b->down * r;
        for (int c = 0; c < loop_cols; c++) {
            *cur_res = s * (*cur_a) * (*cur_b);
            cur_res += res->right;
            cur_a += a->right;
            cur_b += b->right;
        }
    }

    return res;
}

Mat* mat_hadamard_prod(Mat* res, Mat* a, Mat* b, float s) {
    if (a->t || b->t) return mat_hadamard_prod_tr_supp(res, a, b, s);

    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    if (res == NULL) {
        res = mat_malloc(a->rows, a->cols);
    } else {
        assert(res->rows == b->rows);
        assert(res->cols == b->cols);
    }

    float* a_data = a->data;
    float* b_data = b->data;
    float* res_data = res->data;
    for (int i = 0; i < a->rows * a->cols; i++) {
        res_data[i] = s * a_data[i] * b_data[i];
    }
    return res;
}

Mat* mat_hadamard_div_tr_supp(Mat* res, Mat* a, Mat* b, float s) {
    if (a->t == b->t) {
        assert(a->rows == b->rows);
        assert(a->cols == b->cols);
    } else {
        assert(a->rows == b->cols);
        assert(a->cols == b->rows);
    }
    
    if (res == NULL) {
        if (!a->t)
            res = mat_malloc(a->rows, a->cols);
        else
            res = mat_malloc(a->cols, a->rows);
    } else {
        if (a->t == res->t) {
            assert(res->rows == a->rows);
            assert(res->cols == a->cols);
        } else {
            assert(res->rows == a->cols);
            assert(res->cols == a->rows);
        }
    }

    int loop_rows = res->t ? res->cols : res->rows;
    int loop_cols = res->t ? res->rows : res->cols;

    for (int r = 0; r < loop_rows; r++) {
        float* cur_res = res->data + res->down * r;
        float* cur_a = a->data + a->down * r;
        float* cur_b = b->data + b->down * r;
        for (int c = 0; c < loop_cols; c++) {
            *cur_res = s * (*cur_a) / (*cur_b);
            cur_res += res->right;
            cur_a += a->right;
            cur_b += b->right;
        }
    }

    return res;
}

Mat* mat_hadamard_div(Mat* res, Mat* a, Mat* b, float s) {
    if (a->t || b->t) return mat_hadamard_div_tr_supp(res, a, b, s);

    assert(a->rows == b->rows);
    assert(a->cols == b->cols);
    if (res == NULL) {
        res = mat_malloc(a->rows, a->cols);
    } else {
        assert(res->rows == b->rows);
        assert(res->cols == b->cols);
    }

    float* a_data = a->data;
    float* b_data = b->data;
    float* res_data = res->data;
    for (int i = 0; i < a->rows * a->cols; i++) {
        res_data[i] = s * a_data[i] / b_data[i];
    }
    return res;
}

Mat* mat_element_invert(Mat* res, Mat* a) {
    if (res == NULL) {
        if (!a->t)
            res = mat_malloc(a->rows, a->cols);
        else
            res = mat_malloc(a->cols, a->rows);
    } else {
        assert((res->rows == a->rows && res->cols == a->cols) || (res->rows == a->cols && res->cols == a->rows));
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        *(res->data + i) = 1 / *(a->data + i);
    }
    return res;
}

Mat* mat_transpose_old(Mat* res, Mat* a) {
    assert(a != res);
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

float mat_min(Mat* m) {
    float min = m->data[0];
    for (int i = 1; i < m->rows * m->cols; i++)
        if (m->data[i] < min) min = m->data[i];
    return min;
}

const char __mat_shades[5] = {'.', '-', 'o', '#', '@'};

void mat_print_shades(Mat* m, float black, float white) {
    float* m_data = m->data;

    for (int i = 0; i < m->rows * m->cols; i++) {
        float normalized;
        if (white - black != 0) normalized = (m_data[i] - black) / (white - black);
        else normalized = 0;
        int shade_idx = (int)round(normalized * (float)(5 - 1));
        char c = __mat_shades[shade_idx];
        printf("%c", c);
        if (i % m->cols == m->cols - 1)
            printf("\n");
    }
}