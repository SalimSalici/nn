#ifndef _MAT_H
#define _MAT_H

#define MAT_ELEM_IDX(m, row, col) ((row) * ((m)->cols) + (col))

typedef struct Mat {
    int rows;
    int cols;
    float* data;
} Mat;

float mat_standard_norm_filler_cb(float cur, int row, int col, void* func_args);
float mat_norm_filler_cb(float cur, int row, int col, void* func_args);
float mat_zero_filler_cb(float cur, int row, int col, void* func_args);
Mat* mat_malloc(int rows, int cols);
Mat* mat_malloc_cpy(int rows, int cols, float cpy[rows][cols]);
Mat* mat_cpy(Mat* m);
void mat_free(Mat* m);
Mat* mat_mult(Mat* res, Mat* a, Mat* b);
Mat* mat_add(Mat* res, Mat* a, Mat* b);
Mat* mat_sub(Mat* res, Mat* a, Mat* b);
Mat* mat_hadamard_prod(Mat* res, Mat* a, Mat* b);
Mat* mat_transpose(Mat* res, Mat* a);
Mat* mat_scale(Mat* res, Mat* a, float f);
Mat* mat_fill(Mat* m, float f);
Mat* mat_fill_func(Mat* res, Mat* m, float (*f)(float, int, int, void*), void* func_args);
Mat* mat_diag_fill(Mat* m, float f);
float mat_max(Mat* m);
void mat_print(Mat* m);

#endif