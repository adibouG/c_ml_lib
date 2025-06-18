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

typedef int Errno;

typedef struct {
    size_t count;
    size_t capacity;
    float * items;
} Chart;

typedef struct {
    Vector2 point;
    Vector2 size; // x = w  y = h
} Coords;

//Dynamic array implrementation using a macro
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


#define IMG_FACTOR 80
#define IMG_WIDTH (16 * (IMG_FACTOR))
#define IMG_HEIGHT (9 * (IMG_FACTOR))

#define IMG_PADDING 5
#define FPS 60

const size_t TRAINING_DEFAULT_ITER = 50 * 1000; // 50k iterations by default
const size_t TRAINING_MIN_ITER = 5 * 1000; // 5k iterations minimum
const size_t TRAINING_MAX_ITER = 1000 * 1000; // 1M iterations maximum
const float_t TRAINING_DEFAULT_DIFF_DELTA = 1e-1f;
const float_t TRAINING_DEFAULT_LEARN_RATE = 1.f;
const size_t UI_WIDGET_LAYOUT = 2;
const size_t UI_FPS = 60;
const size_t UI_IMG_PADDING = 5;
const Color IMG_BG_COLOR = BLACK;
const Color IMG_FG_COLOR = GRAY;
const Color COLOR_LOW_ACTIVITY = BLUE;
const Color COLOR_HIGH_ACTIVITY = GREEN;

void nn_render(NN nn, int pos_x, int pos_y, int w, int h) 
{
    size_t arch_count = nn.count + 1;
    float  nn_layer_pad_h = w / arch_count;

    size_t resiz_factor = h > w ? h : w;

    float NEURON_RADIUS = 10 + (h/50) ;
    float CONX_WIDTH = 2 + ((h)/100)/250;

    Color activate = COLOR_HIGH_ACTIVITY;
    Color deactivate = COLOR_LOW_ACTIVITY;
    // TODO: 
    //  - extract element values for hovering displays
    
                    
    for (size_t l=0; l < arch_count; ++l) {

        int layer_pad_v1 = h / nn.as[l].cols;

        for (size_t i=0; i < nn.as[l].cols; ++i) {

            int cx1 = pos_x + nn_layer_pad_h * l + nn_layer_pad_h  / 2;
            int cy1 = pos_y + layer_pad_v1 * i + layer_pad_v1 / 2;

            // neuron inter layers connection network
            if (l + 1 < arch_count)
            {
                int layer_pad_v2 = h / nn.as[l+1].cols;

                for (size_t j=0; j < nn.as[l+1].cols; ++j) {

                    int cx2 = pos_x + nn_layer_pad_h * (l+1) + nn_layer_pad_h/2;
                    int cy2 = pos_y + layer_pad_v2 * j + layer_pad_v2/2;
                    // draw connection line between neurons
                    float weight_value = sigmf(MAT_AT(nn.ws[l], j, i));
                    float alpha = floorf(255.f * weight_value);
                    activate.a = alpha;
                    float thick = CONX_WIDTH + floorf(alpha/153);
                    Color dynamic_color = ColorAlphaBlend(deactivate, activate, WHITE);
                    Vector2 st = { cx1, cy1 };
                    Vector2 en = { cx2, cy2 };
                    DrawLineEx (st, en, thick, dynamic_color);
                } // __ for __
            } // __ if __

            if (l > 0)  // internal layer neurons
            {

                float bias_value = sigmf(MAT_AT(nn.bs[l-1], 0, i));
                float alpha = floorf(255.f * bias_value);
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
    *min = FLT_MAX ;
    *max = FLT_MIN ;
    for (size_t i=0; i < chart.count; ++i) {
        if (*max < chart.items[i]) *max = chart.items[i];
        if (*min > chart.items[i]) *min = chart.items[i];
    }
}


void cost_graph_render (Chart chart, int xpos, int ypos, int img_w, int img_h)
{
    float min, max;
    float PADDING_INNER_LEFT = img_w * 0.05f;
    float PADDING_INNER_RIGHT = PADDING_INNER_LEFT;
    Color color_chart = RED;
    Color color_ref_y0 = WHITE;
    size_t n = chart.count;
    if (chart.items == NULL || chart.count == 0) {
        DrawText("No data to display", xpos + 5, ypos + 5, img_h*0.04, color_ref_y0);
        return;
    }
    if (chart.count < 2) {
        DrawText("Not enough data to display", xpos + 5, ypos + 5, img_h*0.04, color_ref_y0);
        return;
    }
   // chart_min_max (&chart, *min, *max); inlined to simplify and  avoid dereferencing
    min = FLT_MAX ;
    max = FLT_MIN ;
    for (size_t i=0; i < chart.count; ++i) {
        if (max < chart.items[i]) max = chart.items[i];
        if (min > chart.items[i]) min = chart.items[i];
    }
   //__chart_min_max: end 
    if (min > 0) min = 0;
    //if (min > max) min = max;
    if (n < 1000) n = 1000;   
    
    char buffer_cost_text_value[64];

    for (size_t i=0; i+1 < chart.count; ++i) {

        float x1 = xpos + (float) img_w / n*(i);
        float y1 = ypos + (1 - (chart.items[i] - min) / (max - min)) * img_h;
        
        float x2 = xpos + (float) img_w / n*(i+1);
        float y2 = ypos + (1 - (chart.items[i+1] - min) / (max - min)) * img_h;
        // chart line-points for plotting cost trace
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, img_h*0.004, color_chart);


        // TODO: dynamic chart value y axis ref line and text 
        /*
        DrawLineEx((Vector2){xpos + 0, y1}, (Vector2){x1, y1}, img_h*0.004, color_ref_y0);
        
        if (cost_value_display == FOLLOW_ON_AXIS || cost_value_display == FOLLOW_ON_CHART) {
            snprintf(buffer_cost_text_value, sizeof(buffer_cost_text_value), "%f", chart.items[i]);
            if (chart.count > 0) {
                snprintf(buffer_cost_text_value, sizeof(buffer_cost_text_value), "%f", chart.items[chart.count - 1]);
                DrawText(buffer, xpos + 0, ypos + 0, img_h*0.04, color_ref_y0);
            //  DrawText("Cost: %f", x1, ref_y0 - img_h*0.04, img_h*0.04, color_ref_y0); 
            }
        }
        */
    }

    // draw chart reference lines
    // horizontal X axis reference line at y = 0 <=> Cost = 0
    float ref_y0 = ypos + (1 - (0 - min) / (max - min)) * img_h;
    DrawLineEx((Vector2){xpos + 0, ref_y0}, (Vector2){xpos + img_w - 1, ref_y0}, img_h*0.004, color_ref_y0);
    DrawText("Cost: 0", img_w*0.04 - xpos, ref_y0 - 2*img_h*0.04, img_h*0.04, color_ref_y0);     	

    // TODO : vertical Y axis reference for cost dynamic value display
    /*
    float ref_x0 = ypos + (1 - (0 - min) / (max - min)) * img_h;
    DrawLineEx((Vector2){xpos + 0, ref_y0}, (Vector2){xpos + img_w - 1, ref_y0}, img_h*0.004, color_ref_y0);
    DrawText("Cost:0", img_w + xpos, ref_y0 - img_h*0.04, img_h*0.04, color_ref_y0); 
    */

    if (chart.count > 0) {
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%f", chart.items[chart.count - 1]);
        DrawText(buffer, xpos + 0, ypos + 0, img_h*0.04, color_ref_y0);
    }
}


void displayOriginalImage(int img_w, int img_h, Mat to, bool hide_null)
{
    printf("*** 'ASCII styled': Original image pixel data: ***\n");
    
    for (size_t y = 0; y < (size_t) img_h; ++y){
        for (size_t x = 0; x < (size_t) img_w; ++x){
            size_t idx = y * img_w + x;
            uint8_t pixel = MAT_AT(to, idx, 0)*255.f;
            
            (hide_null && pixel == 0) ?
                printf("    ") :
                printf("%3u ", pixel);
        }
        printf("\n");
    }      
}

void displayTrainingResult(NN nn, int img_w, int img_h, bool hide_null)
{
    printf("*** 'ASCII styled': NN Post-Training Result image pixel data: ***\n");

    for (size_t y=0; y < (size_t)img_h; ++y){
        for (size_t x=0; x < (size_t)img_w ; ++x){
            MAT_AT(NN_IN(nn), 0, 0) = (float) x / (28 - 1);
            MAT_AT(NN_IN(nn), 0, 1) = (float) y / (28 - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUT(nn), 0, 0)*255.f;

            (hide_null && pixel == 0) ?
                printf("    ") :
                printf("%3u ", pixel);
        }
        printf("\n");
    }
}


/*
 * TODO: 
 *  - add dynamic chart value display
 *  - handle in-training setting manual update issues:
 *      -- max iteration overflow .
 *  - add weight/bias/... value display on neuron hovering
 *  - add display of activity values on element hovering
 *  - add/try different options for dynamic cost value display on charts
 *  - add dynamic learn rate 
 *
 */
NN displayTrainingUI(NN nn, NN g, Mat ti, Mat to, float learn_rate, size_t ITER,  char * title )
{
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(IMG_WIDTH, IMG_HEIGHT, title);
    SetTargetFPS(FPS);

    Chart cost_trace = {0};
    int num_of_widgets = UI_WIDGET_LAYOUT; // 2 widget as default: neuron network and cost trace
    size_t iter = 0;
    bool stop_training = true;
    while (!WindowShouldClose())
    {
        // SPACE: start/pause training
        if (IsKeyPressed(KEY_SPACE)) {
            stop_training = !stop_training;
        }
        // ESC: stop & exit training UI
        else if (IsKeyPressed(KEY_ESCAPE)) {
            CloseWindow();
            return nn;
        }
        // R: reset/restart current training session
        else if (IsKeyPressed(KEY_R)) {
            stop_training = true;
            // reset training
            nn_rand(nn, -1, 1);
            nn_zero(g);
            //da_append(&cost_trace, 0.f);
            iter = 0;
            cost_trace.count = 0;
        }
        // UP/DOWN: increase/decrease learn rate
        else if (IsKeyPressed(KEY_UP)) {
            learn_rate += 0.1f;
            if (learn_rate > 10.f) learn_rate = 10.f;
        }
        else if (IsKeyPressed(KEY_DOWN)) {
            learn_rate -= 0.1f;
            if (learn_rate < 0.0001f) learn_rate = 0.0001f;
        }
        // LEFT/RIGHT: decrease/increase max iterations
        else if (IsKeyPressed(KEY_LEFT) && ITER > TRAINING_MIN_ITER) {
            ITER -= TRAINING_MIN_ITER;
            if (ITER < TRAINING_MIN_ITER) ITER = TRAINING_MIN_ITER;
        }
        else if (IsKeyPressed(KEY_RIGHT) && ITER < TRAINING_MAX_ITER) {
            ITER += TRAINING_MIN_ITER;
            if (ITER > TRAINING_MAX_ITER) ITER = TRAINING_MAX_ITER;
        }
        float cost = nn_cost(nn, ti, to);
        // nn_finite_diff(nn, g, diff_delta, ti, to);
        if (iter > ITER) {
            stop_training = true;
        }
        
        if (!stop_training) {
        
            if (iter >= 0 && iter < ITER) {
                iter += 1;
                nn_backprop(nn, g, ti, to);
                nn_learn(nn, g, learn_rate);
                da_append(&cost_trace, cost); 
            }
        }
        BeginDrawing();
        ClearBackground(IMG_BG_COLOR);
        int w = GetRenderWidth();
        int h = GetRenderHeight();
        int vpad = IMG_PADDING * w/100;
        int hpad = IMG_PADDING * h/100;
        
        int graph_w  =  (w-2*vpad)/num_of_widgets;

        int graph_h =   (h-2*hpad)*3/4;
        int graph_pos_y = h/2 - graph_h/2;
        int graph_pos_x ; // xpos;
        {
            graph_pos_x = w - vpad - graph_w; // xpos;
            nn_render(nn, graph_pos_x, graph_pos_y, graph_w, graph_h);
            graph_pos_y = h/2 - graph_h/2;
            graph_pos_x = vpad; // xpos;
            cost_graph_render(cost_trace, graph_pos_x, graph_pos_y, graph_w, graph_h);
            
            char buffer[256];
            snprintf(buffer, sizeof(buffer),
                   "Iterations: %zu/%zu - LearnRate: %f - Cost: %f",
                   iter, ITER, learn_rate, cost);
            DrawText(buffer, 0, 0, h*0.04, WHITE);
        }
        EndDrawing();
        
    }
    CloseWindow();
    return nn;
}


int main(int argc, char * argv[])
{
    srand(time(0));

     if (argc < 3) {
        fprintf(stderr, "missing arguments 1 and/or 2:\n");
        fprintf(stderr, "usage: %s <archfile.arch> <datafile.mat>\n", argv[0]);
        return 1;
    }

    char * arch_file_name = argv[1];
    char * data_file_name = argv[2];
    
    char * out_rescaled_file_name = "upscaled.out.png";
    if (argc == 4) {
        out_rescaled_file_name = argv[3];
    }
    
    int out_rescaled_img_size_w = 512;
    int out_rescaled_img_size_h = 512;
    if (argc == 5 || argc == 6) {
        out_rescaled_img_size_w = atoi(argv[4]);
        if (argc == 6) {
            out_rescaled_img_size_h = atoi(argv[5]);
        }
        else {
            out_rescaled_img_size_h = out_rescaled_img_size_w;
        }
    }
    

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

    Mat t = mat_load(data_file_name)  ;
    ML_ASSERT(arch.count > 1);
  
    size_t arch_in_sz = arch.items[0]    ;
    size_t arch_out_sz = arch.items[arch.count - 1]    ;
    ML_ASSERT(t.cols == arch_in_sz + arch_out_sz);
   
    NN nn = nn_alloc(arch.items, arch.count);

     Mat ti = {
        .es = &MAT_AT(t, 0, 0),
        .cols = arch_in_sz,
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
    
    char * title = "ML image" ;
    char * meth_name = "back_prop";

    float diff_delta = 1.f; //1e-1;
    // TO DO: make the learning rate dynamic
    float learn_rate = 1.f; //0.1f; 

    size_t ITER = TRAINING_DEFAULT_ITER;
    
    nn = displayTrainingUI(nn, g, ti, to, learn_rate, ITER, title);
   
    size_t img_h = sqrt(t.rows) ;
    size_t img_w = img_h;
    //TODO: HANDLE ASPECT RATIO for non-square images
    // TO DO: handle non-square images WITH ASPECT RATIO FOR UPSCALING

    if (img_w * img_h != t.rows) {
        fprintf(stderr, "image size is not square: %zu x %zu\n", img_w, img_h);
        return 1;
    }

    displayOriginalImage(img_w, img_h, to, false);
    displayTrainingResult(nn, img_w, img_h, false);
    
    size_t img_out_w = (size_t) out_rescaled_img_size_w;
    size_t img_out_h = (size_t) out_rescaled_img_size_h;
 
    //uint8_t * img_out_pixels = (uint8_t *) malloc(img_out_h * img_out_w * sizeof(uint8_t));
    uint8_t * img_out_pixels = malloc(img_out_h * img_out_w * sizeof(*img_out_pixels));
    assert(img_out_pixels != NULL);

    for (size_t y=0; y <  img_out_h; ++y){
        for (size_t x=0; x <  img_out_w; ++x){
            MAT_AT(NN_IN(nn), 0, 0) = (float) x / (img_out_w - 1);
            MAT_AT(NN_IN(nn), 0, 1) = (float) y / (img_out_h - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUT(nn), 0, 0)*255.f;
            img_out_pixels[y * img_out_w + x] = pixel;
        }
    }
 
    stbi_write_png(out_rescaled_file_name, img_out_w, img_out_h, 1, img_out_pixels, 0);
    printf("upscaled image %zu x %zu saved as %s\n", img_out_w, img_out_h, out_rescaled_file_name);

    // save upscaled image
    if (!stbi_write_png(out_rescaled_file_name, img_out_w, img_out_h, 1, img_out_pixels, 0)) 
    {
        fprintf(stderr, "Error: failed to save upscaled image: %s\n", out_rescaled_file_name);
        return 1;
    }
   
    printf("upscaled image %zu x %zu saved as %s\n", img_out_w, img_out_h, out_rescaled_file_name); 
    return 0;
    
    

    /*
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
                                   */
    return 0;
}

