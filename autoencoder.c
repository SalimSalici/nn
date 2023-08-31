#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_loader.h"
#include "mat.h"
#include "nn.h"
#include "sample.h"
#include "helper.h"
#include "openblas_config.h"
#include "cblas.h"

#include "tensor.h"
#include "mec.h"
#include "conv2d.h"
#include "maxpool.h"
#include "cmpl.h"

#include "cnn.h"

#include "raylib.h"

const int screenWidth = 800;
const int screenHeight = 800;
const int N = 28;  // Size of the matrix
const int UPSCALE = 10;

void draw_pair(float* in_data, float* out_data, Vector2 position_in, Vector2 position_out) {

    Color pixels_in[N * N];
    Color pixels_out[N * N];

    Image image_in = { 0 };
    Image image_out = { 0 };

    image_in.width = N;
    image_in.height = N;
    image_in.mipmaps = 1;
    image_in.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

    image_out.width = N;
    image_out.height = N;
    image_out.mipmaps = 1;
    image_out.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    // Update the Raylib image from the matrix
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            float value_in = *in_data;
            float value_out = *out_data;
            // float value_in = (*in_data + 1) / 2;
            // float value_out = (*out_data + 1) / 2;
            value_in = value_in > 1.0 ? 1.0 : value_in;
            value_in = value_in < 0.0 ? 0.0 : value_in;
            in_data++;
            out_data++;
            unsigned char colorValue_in = (unsigned char)(value_in * 255);
            unsigned char colorValue_out = (unsigned char)(value_out * 255);
            pixels_in[y * N + x] = (Color){colorValue_in, colorValue_in, colorValue_in, 255};
            pixels_out[y * N + x] = (Color){colorValue_out, colorValue_out, colorValue_out, 255};
        }
    }

    image_in.data = pixels_in;
    Texture2D texture_in = LoadTextureFromImage(image_in);
    image_out.data = pixels_out;
    Texture2D texture_out = LoadTextureFromImage(image_out);

    // Draw the texture
    BeginDrawing();
    DrawTextureEx(texture_in, position_in, 0, 5.0, WHITE);
    DrawTextureEx(texture_out, position_out, 0, 5.0, WHITE);
    EndDrawing();
    // Unload the texture to avoid memory leak
    UnloadTexture(texture_in);
    UnloadTexture(texture_out);
}

int main(int argc, char* argv[]) {


    // Initialize Raylib
    SetTraceLogLevel(LOG_WARNING);
    InitWindow(screenWidth, screenHeight, "Neural Network Visualization");

    // Create an image and a texture    

    NN* nn = nn_malloc(NN_BCE_LOSS);
    nn_add_layer(nn, layer_malloc(0, 28*28, NN_NONE_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(28*28, 400, NN_SIGMOID_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(400, 100, NN_SIGMOID_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(100, 400, NN_SIGMOID_ACT, 0.0));
    nn_add_layer(nn, layer_malloc(400, 28*28, NN_SIGMOID_ACT, 0.0));
    nn_initialize_xavier(nn);

    float lr = 0.1; // learning rate
    float lambda = 0.00; // L2 regularization
    int epochs = 100;
    int minibatch_size = 50;
    int training_samples_count = 60000;
    int test_samples_count = 1000;
    float black = 0;
    float white = 1;

    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    Sample** training_samples = mnist_samples_to_autoenc_samples(training_data, training_samples_count, black, white);
    Sample** test_samples = mnist_samples_to_autoenc_samples(test_data, test_samples_count, black, white);

    // int idx = 59;
    Mat* in_0 = training_samples[0]->inputs;
    Mat* in_1 = training_samples[1]->inputs;
    Mat* in_2 = training_samples[2]->inputs;

    int cur_epoch = 0;
    // Main training loop
    while (!WindowShouldClose())
    {
        // Train the neural network for one epoch
        // TrainOneEpoch();

        if (cur_epoch < 30) {
            nn_one_epoch(nn, training_samples, training_samples_count, epochs, minibatch_size, lr, lambda, test_samples, test_samples_count);
            cur_epoch++;
        }

        nn_set_layers_group_count(nn, 1);
        nn_set_mode(nn, NN_INFERENCE);

        // Mat* out_0 = mat_cpy(nn_feedforward(nn, in_0));
        // Mat* out_1 = mat_cpy(nn_feedforward(nn, in_1));
        // Mat* out_2 = mat_cpy(nn_feedforward(nn, in_2));
        Mat* out_0 = mat_cpy(nn_feedforward(nn, training_samples[0]->inputs));
        Mat* out_1 = mat_cpy(nn_feedforward(nn, training_samples[1]->inputs));
        Mat* out_2 = mat_cpy(nn_feedforward(nn, training_samples[2]->inputs));

        // float* in_data_0 = in_0->data;
        // float* in_data_1 = in_1->data;
        // float* in_data_2 = in_2->data;
        float* in_data_0 = training_samples[0]->inputs->data;
        float* in_data_1 = training_samples[1]->inputs->data;
        float* in_data_2 = training_samples[2]->inputs->data;
        float* out_data_0 = out_0->data;
        float* out_data_1 = out_1->data;
        float* out_data_2 = out_2->data;

        // BeginDrawing();

        Vector2 position0 = {0, 0};
        Vector2 position1 = {140, 0};
        draw_pair(in_data_0, out_data_0, position0, position1);

        position0.y = 140;
        position1.x = 140;
        position1.y = 140;
        draw_pair(in_data_1, out_data_1, position0, position1);

        position0.y = 280;
        position1.x = 140;
        position1.y = 280;
        draw_pair(in_data_2, out_data_2, position0, position1);

        // EndDrawing();

        // Check for window close events
        if (WindowShouldClose()) {
            break;
        }
    }

    // Cleanup
    CloseWindow();

    return 0;

}