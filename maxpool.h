#ifndef _MAXPOOL_H
#define _MAXPOOL_H

typedef struct Maxpool {
    int n;
    int h_in;
    int w_in;
    int c;

    int mh;
    int mw;
    int stride;

    int h_out;
    int w_out;

    Mat* inputs_d; // (H, N, W, C) tensor format - (H, N*W*C) matrix format
    Mat* outputs; // (N, H, W, C) tensor format - (N*H, W*C) matrix format

    int* stored_indeces;
} Maxpool;

Maxpool* maxpool_malloc(int h_in, int w_in, int c, int mh, int mw, int stride) {
    
    if (!is_divisible(h_in - mh, stride)) {
        printf("Dimensions not compatible... h_in: %d, mh: %d, s: %d", h_in, mh, stride);
        exit(1);
    }

    if (!is_divisible(w_in - mw, stride)) {
        printf("Dimensions not compatible... w_in: %d, mw: %d, s: %d", w_in, mw, stride);
        exit(1);
    }

    Maxpool* maxpool    = (Maxpool*)malloc(sizeof(Maxpool));
    maxpool->n          = 1;
    maxpool->h_in       = h_in;
    maxpool->w_in       = w_in;
    maxpool->c          = c;
    maxpool->mh         = mh;
    maxpool->mw         = mw;
    maxpool->stride     = stride;

    int h_out = (h_in - mh) / stride + 1;
    maxpool->h_out = h_out;
    int w_out = (w_in - mw) / stride + 1;
    maxpool->w_out = w_out;

    maxpool->inputs_d = mat_malloc(h_in, 1 * w_in * c);
    maxpool->outputs = mat_malloc(1 * h_out, w_out * c);

    maxpool->stored_indeces = (int*)malloc(sizeof(int) * 1 * h_out * w_out * c);

    return maxpool;
}

void maxpool_free(Maxpool* maxpool) {
    mat_free(maxpool->inputs_d);
    mat_free(maxpool->outputs);
    free(maxpool->stored_indeces);
    free(maxpool);
}

Maxpool* maxpool_set_n(Maxpool* maxpool, int n) {
    maxpool->n = n;

    mat_free(maxpool->inputs_d);
    mat_free(maxpool->outputs);
    free(maxpool->stored_indeces);

    maxpool->inputs_d = mat_malloc(maxpool->h_in, n * maxpool->w_in * maxpool->c);
    maxpool->outputs = mat_malloc(n * maxpool->h_out, maxpool->w_out * maxpool->c);
    maxpool->stored_indeces = (int*)malloc(sizeof(int) * n * maxpool->h_out * maxpool->w_out * maxpool->c);

    return maxpool;
}

Maxpool* maxpool_forward(Maxpool* maxpool, Mat* inputs) {
    // assert(inputs->rows == maxpool->h_in && inputs->cols == maxpool->n * maxpool->w_in * maxpool->c);

    int N = maxpool->n;
    int C = maxpool->c;
    // int h_in = maxpool->h_in;
    int w_in = maxpool->w_in;
    int h_out = maxpool->h_out;
    int w_out = maxpool->w_out;
    int mh = maxpool->mh;
    int mw = maxpool->mw;
    int stride = maxpool->stride;

    float maxes[C]; 
    int indeces[C];

    float* inputs_data = inputs->data;
    int inputs_cols = inputs->cols;

    float *cur_outputs_data = maxpool->outputs->data;
    int* cur_indeces_data = maxpool->stored_indeces;

    for (int n = 0; n < N; n++) {

        float* start_inputs_n = inputs_data + n * w_in * C;

        for (int h_o = 0; h_o < h_out; h_o++) {

            float* start_inputs = start_inputs_n + h_o * stride * inputs_cols;

            for (int w_o = 0; w_o < w_out; w_o++) {

                float *start_inputs_w = start_inputs + w_o * stride * C;

                for (int i = 0; i < C; i++)
                    maxes[i] = -INFINITY;

                for (int w = 0; w < mw; w++) {

                    float* start_inputs_s = start_inputs_w + w * C;

                    for (int h = 0; h < mh; h++) {
                    
                        start_inputs_s += h * inputs_cols;

                        for (int c = 0; c < C; c++) {
                            if (start_inputs_s[c] >= maxes[c]) {
                                maxes[c] = start_inputs_s[c];
                                indeces[c] = start_inputs_s + c - inputs_data;
                            }
                        }

                    }
                }

                memcpy(cur_outputs_data, maxes, sizeof(float) * C);
                memcpy(cur_indeces_data, indeces, sizeof(int) * C);
                cur_outputs_data += C;
                cur_indeces_data += C;
            }
        }

    }

    return maxpool;
}

Maxpool* maxpool_backward(Maxpool* maxpool, Conv2d* conv2d, Mat* ds) {

    int elements = maxpool->outputs->rows * maxpool->outputs->cols;
    
    mat_fill(conv2d->conv_hnwc_d, 0.0f);
    float* conv_ds = conv2d->conv_hnwc_d->data;

    int* cur_stored_indeces = maxpool->stored_indeces;
    float* cur_ds = ds->data;

    for (int i = 0; i < elements; i++) {
        conv_ds[cur_stored_indeces[i]] += cur_ds[i]; 
    }

    return maxpool;
}

#endif