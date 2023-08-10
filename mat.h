#ifndef _MAT_H
#define _MAT_H

#define MAT_USE_OPENBLAS

#define MAT_ELEM_IDX(m, row, col) ((row) * ((m)->cols) + (col))

typedef struct Mat {
    int rows;
    int cols;
    float* data;
    int t;
    int right;
    int down;
} Mat;

float mat_standard_norm_filler_cb(float cur, int row, int col, void* func_args);
float mat_norm_filler_cb(float cur, int row, int col, void* func_args);
float mat_zero_filler_cb(float cur, int row, int col, void* func_args);
Mat* mat_malloc(int rows, int cols);
Mat* mat_malloc_cpy(int rows, int cols, float cpy[rows][cols]);
Mat* mat_cpy(Mat* m);
Mat* mat_malloc_from_file(int rows, int cols, char* filename);
void mat_free(Mat* m);
int mat_equals(Mat* a, Mat* b);
int mat_equals_bin(Mat* a, Mat* b);
Mat* mat_mult(Mat* res, Mat* a, Mat* b);
Mat* mat_mult_mm(Mat* res, Mat* a, Mat* b, float ab_s, float c_s);
Mat* mat_mult_mv(Mat* res, Mat* a, Mat* b, float ab_s, float c_s);
Mat* mat_mult_old(Mat* res, Mat* a, Mat* b);
Mat* mat_add(Mat* res, Mat* a, Mat* b);
Mat* mat_add_old(Mat* res, Mat* a, Mat* b);
Mat* mat_sub(Mat* res, Mat* a, Mat* b);
Mat* mat_sub_old(Mat* res, Mat* a, Mat* b);
Mat* mat_hadamard_div(Mat* res, Mat* a, Mat* b, float s);
Mat* mat_hadamard_prod(Mat* res, Mat* a, Mat* b, float s);
Mat* mat_hadamard_prod_old(Mat* res, Mat* a, Mat* b);
Mat* mat_element_invert(Mat* res, Mat* a);
Mat* mat_transpose(Mat* a);
Mat* mat_transpose_old(Mat* res, Mat* a);
Mat* mat_scale(Mat* res, Mat* a, float f);
Mat* mat_fill(Mat* m, float f);
Mat* mat_fill_func(Mat* res, Mat* m, float (*f)(float, int, int, void*), void* func_args);
Mat* mat_diag_fill(Mat* m, float f);
float mat_max(Mat* m);
float mat_min(Mat* m);
void mat_print(Mat* m);
void mat_print_shades(Mat* m, float black, float white);

#endif