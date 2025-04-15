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
float * td = td_xor;
size_t  td_count = 4;



int main()
{
    srand(time(0));

    size_t stride = 3;
    size_t len = td_count; //(sizeof(td)/sizeof(&td[0]))/stride;

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
    nn_rand(nn, 0, 1);

    size_t ITER = 5000;

    float delta = 1e-1;
    float learn_rate = 1; //1e-1;

    printf("cost=%f\n", nn_cost(nn, ti, to));

    const Color IMG_BG = RAYWHITE;
    int IMG_WIDTH = 800;
    int IMG_HEIGHT = 600;
    const int IMG_PADDIN_H = 10;
    const int IMG_PADDIN_V = 10;

    int NEURON_DIAMETER = 25;
    int NEURON_PADDIN = 5;
    const Color NEURON_COLOR_FILL_IDL = RED;

    int NEURON_CONX_WIDTH = 2;
    int NEURON_CONX_MAX_WIDTH = 10;
    const Color NEURON_CONX_COLOR_IDL = RAYWHITE;
    const Color NEURON_CONX_COLOR_ACT = RED ;

    int LAYER_GAP_H = 25;
    // not const values Dv are :
    //  set with the above min values and proportionnal to :
    //    - a fixed fit ratio =>
    //    This fixed fit ratio is calculated at 1rst start from:
    //         = IMG_SIZe , the nbr of Layer amd the layer max size
    //     - a zoom factor =>
    //     This zoom factor value is a runtime ratio from :
    //          = modified user zoom setting (-1 to 1 )
    //
    InitWindow(IMG_WIDTH, IMG_HEIGHT, "ML");
    SetTargetFPS(60);

    const int IMG_CENTER_H = IMG_WIDTH / 2;
    const int IMG_CENTER_V = IMG_HEIGHT / 2;
    /*
    int get_center(int v) {
        return v / 2;
    };
    int nbr_of_layers  =
    int in_pos_y = (ImG_HEIGHT - 2 * IMG_PADDIN_V) / arch.
    int in_pos_x =
    void draw_neuron(int x, int y, Color c){
        DrawCircle(x, y, NEURON_DIAMETER, NEURON_COLOR_FILL_IDL);
    };
    void draw_layer(int x, int y, Mat mat) {
        for (size_t r=0; r<mat.rows;++r){
            for (size_t c=0; r<mat.cols;++c){
                draw_neuron(int x, int y, Color c){

     */

    int arch_count = ARRAY_LEN(arch);

    int  nn_w = IMG_WIDTH - IMG_PADDIN_H*2;
    int  nn_h = IMG_HEIGHT - IMG_PADDIN_V*2;
    int  nn_x = IMG_WIDTH/2 - nn_w/2;
    int  nn_y = IMG_HEIGHT/2 - nn_h/2;
    int layer_pad_h = nn_w / arch_count;


    while (!WindowShouldClose()) {
        BeginDrawing();
        {
            ClearBackground(IMG_BG);

    for (size_t l=0; l<arch_count;++l){
        int layer_pad_v1 = nn_h / arch[l];
        for (size_t i=0; i<arch[l];++i){
            int cx1 = nn_x + layer_pad_h *l + layer_pad_h/2;
            int cy1 = nn_y + layer_pad_v1 *i + layer_pad_v1/2;
            DrawCircle(cx1, cy1, NEURON_DIAMETER, NEURON_COLOR_FILL_IDL);
            if (l+1 < arch_count){
                int layer_pad_v2 = nn_h / arch[l+1];
                for (size_t j=0; j<arch[l+1];++j){
                    int cx2 = nn_x + layer_pad_h *(l+1) + layer_pad_h/2;
                    int cy2 = nn_y + layer_pad_v2 *j + layer_pad_v2/2;
                    DrawLine(cx1, cy1, cx2, cy2, NEURON_CONX_COLOR_IDL);
                }
            }
        }
    }

        }
        EndDrawing();
    }

    CloseWindow();

    return 0;

    for (size_t i=0;i<ITER;++i){
#if 0
        nn_finite_diff(nn, g, delta, ti, to);
#else
        nn_backprop(nn, g, ti, to);
#endif
     //   NN_PRINT(g);
        nn_learn(nn, g, learn_rate);
        printf("%zu: cost=%f\n", i, nn_cost(nn, ti, to));
    }
     //   NN_PRINT(nn);

    printf("Post training Model Test\n");
    for (size_t i=0; i < 2; ++i){
        for (size_t j=0; j < 2; ++j){
            MAT_AT(NN_IN(nn), 0, 0) = i;
            MAT_AT(NN_IN(nn), 0, 1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUT(nn), 0, 0));
        }
    }


    return 0;
}
