#include <stdio.h>
#include <float.h>

#include "raylib.h"

#define ML_IMP
#include "mllib.h"

#define SV_IMPLEMENTATION
#include "sv.h"

#ifndef ML_GRAPH_ASSERT
#define ML_GRAPH_ASSERT ML_ASSERT
#endif



#define IMG_BG_COLOR BLACK
#define IMG_FG_COLOR GRAY

#define COLOR_LOW_ACTIVITY  GRAY
#define COLOR_HIGH_ACTIVITY GREEN

typedef int Errno ;

typedef struct {
    size_t count;
    size_t capacity;
    float * items;
} Chart;


typedef struct {
    Vector2 point;
    Vector2 size; // x = w  y = h
} Coords;




#define DA_INIT_CAP 256

#define da_append(da, item) do {                                                    \
if ((da)->count >= (da)->capacity) {                                                \
    (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity*2;          \
    (da)->items = realloc((da)->items, (da)->capacity*sizeof(*(da)->items));        \
    assert((da)->items != NULL &&                                                   \
        "da_append macro: dynamic array fail to realloc");                          \
}                                                                                   \
(da)->items[(da)->count++] = (item);                                                \
}                                                                                   \
while (0)                                                                           \


void nn_render(NN nn, int nn_pos_x, int nn_pos_y, int nn_w, int nn_h) {

    size_t arch_count = nn.count + 1;
    float  nn_layer_pad_h = nn_w / arch_count;

    float NEURON_RADIUS = 10 + (nn_h/50) ;
    float CONX_WIDTH = 2 + ((nn_h)/100)/250;

    Color activate = COLOR_HIGH_ACTIVITY;
    Color deactivate = COLOR_LOW_ACTIVITY;
    for (size_t l=0; l < arch_count; ++l) {

        int layer_pad_v1 = nn_h / nn.as[l].cols;

        for (size_t i=0; i < nn.as[l].cols; ++i) {

            int cx1 = nn_pos_x + nn_layer_pad_h * l + nn_layer_pad_h  / 2;
            int cy1 = nn_pos_y + layer_pad_v1 * i + layer_pad_v1 / 2;

            // neuron inter layers connection network
            if (l + 1 < arch_count)
            {
                int layer_pad_v2 = nn_h / nn.as[l+1].cols;

                for (size_t j=0; j < nn.as[l+1].cols; ++j) {

                    int cx2 = nn_pos_x + nn_layer_pad_h * (l+1) + nn_layer_pad_h/2;
                    int cy2 = nn_pos_y + layer_pad_v2 * j + layer_pad_v2/2;

                    float weigth_value = sigmf(MAT_AT(nn.ws[l], j, i));
                    float alpha = floorf(255.f*weigth_value);
                    activate.a = alpha;
                    float thick = CONX_WIDTH + floorf(alpha/153);
                    Color colr = ColorAlphaBlend(deactivate, activate, WHITE);
                    Vector2 st = { cx1, cy1 };
                    Vector2 en = { cx2, cy2 };
                    DrawLineEx (st, en, thick, colr);
                } // __ for __
            } // __ if __

            if (l > 0)  // internal layer neurons
            {

                float bias_value = sigmf(MAT_AT(nn.bs[l-1], 0, i));
                float alpha = floorf(255.f*bias_value);
                activate.a = alpha;
                // size_t layer_neuron_nbr = nn.as[l-1].cols
                // float layer_size_neuron_radius = (float) (nn_h/layer_neuron_nbr);
                Color colr = ColorAlphaBlend(deactivate, activate, WHITE);
                DrawCircle(cx1, cy1, NEURON_RADIUS, colr);
            }
            else  // input layer neurons
            {
                DrawCircle(cx1, cy1, NEURON_RADIUS, IMG_FG_COLOR);
            }
        }
    }
}

void chart_min_max (Chart chart, float *min, float *max)
{
    *min =FLT_MAX ;
    *max =FLT_MIN ;
    for (size_t i=0; i < chart.count; ++i) {
        if (*max < chart.items[i]) *max = chart.items[i];
        if (*min > chart.items[i])  *min = chart.items[i];
    }
}


void cost_graph_render (Chart chart, int xpos, int ypos, int img_w, int img_h)
{
    float min, max;

    size_t n = chart.count;
    chart_min_max (chart, &min, &max);

    if (n < 100) n = 100;
    if (min > 0) min = 0;
    if (min > max) min = max;

    for (size_t i=1; i < chart.count; ++i) {

        float x = xpos + (float) img_w / chart.count*(i-1);
        float y = ypos + (1 - (chart.items[i-1] - min) / (max - min)) * img_h;
        float x2 = xpos + (float) img_w / chart.count*(i);
        float y2 = ypos + (1 - (chart.items[i] - min) / (max - min)) * img_h;

        DrawLineEx((Vector2){x, y}, (Vector2){x2, y2}, img_h*0.004, RED);
        //DrawCircle(x, y, img_h*0.004, RED);
    }
}

#define IMG_FACTOR 80
#define IMG_WIDTH (16 * (IMG_FACTOR))
#define IMG_HEIGHT (9 * (IMG_FACTOR))

#define IMG_PADDING 10

#define FPS 60

#define LOAD_FILE

int main(int argc, char * argv[])
{
    srand(time(0));

     if (argc < 3) {
        fprintf(stderr, "missing argument arch_file and/or data_file");
    }
    char * arch_file_name = argv[1];
    char * data_file_name = argv[2];

    // TODO :load a nn def from file
    //char * arch_file_name = "adder.arch";
    //char * data_file_name = "adder.mat";
    int buf_length = 0;
    unsigned char * buf = LoadFileData(arch_file_name, &buf_length);

    String_View file_content = sv_from_parts((const char *) buf, buf_length);

    Arch arch = {0};
    file_content = sv_trim_left(file_content);

    while (file_content.count > 0 && isdigit(file_content.data[0])) {
        size_t arch_data = sv_chop_u64(&file_content) ;
        da_append(&arch, arch_data);
        file_content = sv_trim_left(file_content);
        printf("%zu\n", arch_data);
    }

    Mat  t = mat_load(data_file_name)  ;
    ML_ASSERT(arch.count > 1);
    MAT_PRINT(t);
    size_t arch_in_sz = arch.items[0]    ;
    size_t arch_out_sz = arch.items[arch.count - 1]    ;
    ML_ASSERT(t.cols == arch_in_sz + arch_out_sz);
    NN nn = nn_alloc(arch.items, arch.count);

     Mat ti = {
        .es = &MAT_AT(t, 0, 0),
        .cols =arch_in_sz,
        .rows = t.rows,
        .stride = t.stride
    }; //  mat_alloc(td_rows, 2*BITS);

    Mat to =  {
        .es = &MAT_AT(t, 0, arch_in_sz),
        .cols = arch_out_sz,
        .rows = t.rows,
        .stride = t.stride
    }; //mat_alloc(td_rows, BITS+1);

    NN g = nn_alloc(arch.items, arch.count);
    nn_rand(nn, -1, 1);
    NN_PRINT(nn);
              //  return  0 ;
    char * title = "ML adder" ;
    char * meth_name = "back_prop";

    float diff_delta = 1.f; //1e-1;
    float learn_rate = 1e-1; //1.f

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, title);
    SetTargetFPS(FPS);

    Chart cost_trace = {0};

    size_t ITER = 10000;
    size_t iter = 0;
    while (!WindowShouldClose())
    {
        float cost =  nn_cost(nn, ti, to);
        if (iter < ITER) {
            nn_backprop(nn, g, ti, to);
            nn_learn(nn, g, learn_rate);
            iter += 1;
            cost = nn_cost(nn, ti, to);
            da_append(&cost_trace,cost); // nn_cost(nn, ti, to));

        }
        BeginDrawing();
        ClearBackground(IMG_BG_COLOR);
        int w = GetRenderWidth();
        int h = GetRenderHeight();

        int graph_w  =  w/2 - IMG_PADDING*2/w ;
        int graph_h =   h*3/4 - IMG_PADDING*2/h  ;
        int graph_pos_y = h/2 - graph_h/2;
        int graph_pos_x ; // xpos;
        {
            graph_pos_x = w - graph_w; // xpos;
           // nn_render(nn, nn_coord);
            nn_render(nn, graph_pos_x, graph_pos_y, graph_w, graph_h);

            graph_pos_y = h/2 - graph_h/2;
            graph_pos_x = 0;
            //cost_graph_render(cost_graph, cost_coord);
            cost_graph_render(cost_trace, graph_pos_x, graph_pos_y, graph_w, graph_h);

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Iteration: %zu/%zu - learn_rate: %f - Cost: %f", iter, ITER, learn_rate, cost); //nn_cost(nn, ti, to));
            DrawText(buffer, 0, 0, h*0.04, WHITE);
        }
        EndDrawing();
    }

    CloseWindow();

    size_t BITS = NN_OUT(nn).cols - 1;
    size_t n = (1<<BITS);
    size_t fail = 0;
    for (size_t x = 0; x < n; ++x) {

        for (size_t y = 0; y < n; ++y) {

            size_t z = x + y;

            printf("%zu + %zu = ", x, y);

            for (size_t j = 0; j < BITS; ++j) {

                MAT_AT(NN_IN(nn), 0, y) = (x>>j)&1;
                MAT_AT(NN_IN(nn), 0, y + BITS) = (y>>j)&1;
            }
            nn_forward(nn);

            if (MAT_AT(NN_OUT(nn), 0 , BITS) > 0.5f) {
                if (z >= n) {
                    printf("[OVERFLOW<%zu>]%zu\n", n, z);
                    fail += 1;
                }
            } else {
                size_t cary = 0;
                for (size_t j = 0; j <= BITS; ++j) {
                    size_t bit = MAT_AT(NN_OUT(nn), 0, j) > 0.5f;
                    cary |= bit<<j;
                }
                if (z != cary) {
                    printf("%zu<>%zu\n", z, cary);
                    fail += 1;
                }
            }
        }
    }
    if (fail == 0) printf("OK\n");

    return 0;
}

