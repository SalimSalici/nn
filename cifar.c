#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include "cifar10_loader.h"
#include "mat.h"
#include "sample.h"
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"
#include "cnn.h"

const char* classes[] = {
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
};

Mat* tensor_im2col(Mat* input, int c_out, int h_out, int w_out, int stride) {

}

int main(int argc, char* argv[]) {
    
    srand(time(NULL) * time(NULL));

    goto_set_num_threads(1);
    openblas_set_num_threads(1);

    int training_samples_count = 30;
    // int test_samples_count = 10;
    float black = 0;
    float white = 1;

    Cifar10Sample* training_data = cifar10_load_samples("data/cifar10/data_batch_1.bin", 0, training_samples_count, black, white);
    // MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    for (int x = 0; x < 30; x++) {
        cifar10_print_image(training_data[x].data);
        printf("%s\n\n", classes[training_data[x].label]);
    }
    return 0;
}