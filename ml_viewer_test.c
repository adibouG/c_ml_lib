#include <stdio.h>

#include "raylib.h"

#define ML_IMP
#include "./mllib.h"


float td_or[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,1,
};
float td_and[] = {
    0,0,0,
    0,1,0,
    1,0,0,
    1,1,1,
};
float td_nand[] = {
    0,0,1,
    0,1,1,
    1,0,1,
    1,1,0,
};
float td_xor[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

float * td = td_or;

size_t  td_count = 4;


    const Color IMG_BG = BLACK;
    int IMG_WIDTH = 800;
    int IMG_HEIGHT = 600;

    const int IMG_PADDIN_H = 10;
    const int IMG_PADDIN_V = 10;

    int NEURON_DIAMETER = 25;

    int CONX_WIDTH = 2;
    int CONX_MAX_WIDTH = 10;



    // No weight = REYWHITE
    // weight = 0 RED
    // weight =  1  GREEN
    // Connx =  0  <  1  <  2


// Color ColorAlpha(Color color, float alpha); // Get color with alpha applied, alpha goes from 0.0f to 1.0f
// Color ColorAlphaBlend(Color dst, Color src, Color tint); // Get src alpha-blended into dst color with tint
// Color ColorLerp(Color color1, Color color2, float factor); // Get color lerp interpolation between two colors, factor [0.0f..1.0f]
/*
void draw_neuron(int x, int y, float w){
    Color c = ColorLerp(NEURON_COLOR_FILL_IDL, COLOR_ACT, w);
    DrawCircle(x, y, NEURON_DIAMETER, c);
};

 // void DrawLineEx(Vector2 startPos, Vector2 endPos, float thick, Color color);
void draw_conn_weight(int cx1, int cy1, int cx2, int cy2, float w){
    Color c = ColorLerp(CONX_COLOR_IDL, COLOR_ACT, w);
    DrawLine(cx1, cy1, cx2, cy2, c);
};
*/
void nn_render(NN nn)
{

    Color NEURON_COLOR_FILL_IDL = RAYWHITE;
    Color COLOR_ACT_LOW = GREEN ;
    Color COLOR_ACT_HIGH = RED ;


    int arch_count = nn.count + 1; //ARRAY_LEN(arch);

    int  nn_w = IMG_WIDTH - IMG_PADDIN_H*2;
    int  nn_h = IMG_HEIGHT - IMG_PADDIN_V*2;
    int  nn_x = IMG_WIDTH/2 - nn_w/2;
    int  nn_y = IMG_HEIGHT/2 - nn_h/2;
    int layer_pad_h = nn_w / arch_count;


    //while (!WindowShouldClose())
    {
      //  BeginDrawing();
        {
            ClearBackground(IMG_BG);

            for (size_t l=0; l < arch_count; ++l) {

                int layer_pad_v1 = nn_h / nn.as[l].cols;

                for (size_t i=0; i < nn.as[l].cols; ++i) {

                    int cx1 = nn_x + layer_pad_h *l + layer_pad_h/2;
                    int cy1 = nn_y + layer_pad_v1 *i + layer_pad_v1/2;

                    // connexion
                    if (l+1 < arch_count) {

                        int layer_pad_v2 = nn_h / nn.as[l+1].cols;

                        for (size_t j=0; j < nn.as[l+1].cols; ++j) {

                            int cx2 = nn_x + layer_pad_h *(l+1) + layer_pad_h/2;
                            int cy2 = nn_y + layer_pad_v2 *j + layer_pad_v2/2;

                            int alpha = floorf(255.f*sigmf(MAT_AT(nn.ws[l], j, i)));
                            COLOR_ACT_HIGH.a = alpha;
                            printf("MAT_AT(nn.ws[%zu], %zu, %zu) = %f , connx alpha = %d\n", l, j, i, MAT_AT(nn.ws[l], j, i), alpha);
                            Color colr = ColorAlphaBlend(COLOR_ACT_LOW, COLOR_ACT_HIGH, WHITE);
                            //Color colr = ColorLerp(COLOR_ACT_LOW, COLOR_ACT_HIGH, alpha);
                            DrawLine(cx1, cy1, cx2, cy2, colr);
                        } // __ for __
                    } // __ if __

                    // neuron
                    if (l>0){
                        int alpha = floorf(255.f*sigmf(MAT_AT(nn.bs[l-1], 0, i)));
                        COLOR_ACT_HIGH.a = alpha;
                        printf("MAT_AT(nn.ws[%zu], 0, %zu) = %f , alpha = %d\n", l-1, i, MAT_AT(nn.ws[l-1], 0, i), alpha);

                        Color colr = ColorAlphaBlend(COLOR_ACT_LOW, COLOR_ACT_HIGH, WHITE);
                        //Color colr = ColorLerp(CONX_COLOR_IDL, COLOR_ACT_LOW, alpha);
                        DrawCircle(cx1, cy1, NEURON_DIAMETER, colr);
                    }
                    else {
                        //input layer
                        DrawCircle(cx1, cy1, NEURON_DIAMETER, NEURON_COLOR_FILL_IDL);
                    }
                }
            }
        }
     //   EndDrawing();
    }
    //CloseWindow();
}

int main()
{
    srand(time(0));

    size_t stride = 3;
    size_t len = td_count; //(sizeof(td)/sizeof(&td[0])) /stride;

    Mat ti = {
        .rows = len,
        .cols = 2,
        .stride = stride,
        .es = td
        };

    Mat to = {
        .rows = len,
        .cols = 1,
        .stride = stride,
        .es = td + 2,
        };


    size_t arch[] = {2, 2, 1};

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    size_t ITER = 5000;

    float learn_rate = 1; //1e-1;
    float delta = 1e-1;


    int arch_count = nn.count + 1; //ARRAY_LEN(arch);

    int  nn_w = IMG_WIDTH - IMG_PADDIN_H*2;
    int  nn_h = IMG_HEIGHT - IMG_PADDIN_V*2;
    int  nn_x = IMG_WIDTH/2 - nn_w/2;
    int  nn_y = IMG_HEIGHT/2 - nn_h/2;
    int layer_pad_h = nn_w / arch_count;

    if (1) {
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "ML");
    SetTargetFPS(60);

    size_t it = 0;
    while (!WindowShouldClose())
    {
        if (it < ITER) {
//            nn_render(nn);
            if (1)
                nn_finite_diff(nn, g, delta, ti, to);
            else {
                nn_backprop(nn, g, ti, to);
            }
            nn_learn(nn, g, learn_rate);
            printf("%zu\n", it);
            NN_PRINT(nn);

            it += 1;
        }
        BeginDrawing();
        ClearBackground(IMG_BG);
        nn_render(nn);

        EndDrawing();
    }

    //CloseWindow();
    }

    //printf("************************\n");
    //printf("Post training Model Test\n");
    //printf("************************\n");
    //for (size_t i=0; i < 2; ++i){
        //for (size_t j=0; j < 2; ++j){
            //MAT_AT(NN_IN(nn), 0, 0) = i;
            //MAT_AT(NN_IN(nn), 0, 1) = j;
            //nn_forward(nn);
            //printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUT(nn), 0, 0));
        //}
    //}
    //printf("************************\n");


    //return 0;
#if 0
    for (size_t i=0; i < ITER; ++i) {
        nn_finite_diff(nn, g, delta, ti, to);
        nn_learn(nn, g, learn_rate);
        printf("%zu: cost=%f\n", i, nn_cost(nn, ti, to));
    }
#else
    for (size_t i=0; i < ITER; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, learn_rate);
        printf("%zu: cost=%f\n", i, nn_cost(nn, ti, to));
    }
#endif

    printf("************************\n");
    printf("************************\n");
    NN_PRINT(g);
    printf("************************\n");
    printf("Post training Model Test\n");
    printf("************************\n");
    for (size_t i=0; i < 2; ++i){
        for (size_t j=0; j < 2; ++j){
            MAT_AT(NN_IN(nn), 0, 0) = i;
            MAT_AT(NN_IN(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUT(nn), 0, 0));
        }
    }
    printf("************************\n");

    CloseWindow();
    return 0;
}
