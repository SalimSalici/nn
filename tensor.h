#ifndef _TENSOR_H
#define _TENSOR_H

typedef struct Tensor {
    int B;
    int C;
    int W;
    int H;
    float* data;
} Tensor;

Tensor* tensor_malloc(int b, int c, int w, int h) {
    Tensor* tens = (Tensor*)malloc(sizeof(Tensor));
    tens->B = b;
    tens->C = c;
    tens->W = w;
    tens->H = h;
    tens->data = (float*)malloc(sizeof(float) * b * c * w * h);
    return tens;
}

void tensor_free(Tensor* t) {
    free(t->data);
    free(t);
}

#endif