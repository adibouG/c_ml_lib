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
    size_t type_size; // size of the type of the items in the array
    // items is a pointer to the array of items
    void * items;
} DArray;



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


const size_t UI_FPS = 60;
const size_t UI_WINDOW_FACTOR = 80;
const size_t UI_WINDOW_WIDTH = (16 * (UI_WINDOW_FACTOR));
const size_t UI_WINDOW_HEIGHT = (9 * (UI_WINDOW_FACTOR));

const size_t UI_WINDOW_PADDING = 5;
const size_t TRAINING_DEFAULT_ITER = 50 * 1000; // 50k iterations by default
const size_t TRAINING_MIN_ITER = 5 * 1000; // 5k iterations minimum
const size_t TRAINING_MAX_ITER = 1000 * 1000; // 1M iterations maximum
const float_t TRAINING_DEFAULT_DIFF_DELTA = 1e-1f;
const float_t TRAINING_DEFAULT_LEARN_RATE = 1.f;
const size_t UI_WIDGET_LAYOUT_BASE = {1, 3}; // 1 row and 3 cols : plot, metwork, data
const size_t UI_WIDGET_LAYOUT_LIVE_IMG = {1, 2}; // 2 img side by side: // original and live preview
const size_t UI_WIDGET_LAYOUT_LIVE_IMG_MERGE = {2, 3}; //row1 => in1: img1, coef rate modif, in2: img2 // row2 => out1: merged_img;
const size_t UI_WIDGET_PADDING = 5;
const Color UI_BG_COLOR = GRAY; 
const Color UI_FG_COLOR = WHITE;
const Color UI_HOVER_COLOR = LIGHTGRAY;
const Color COLOR_LOW_ACTIVITY = SKYBLUE;
const Color COLOR_HIGH_ACTIVITY = GREEN;

#define RENDER_WIDTH 512
#define RENDER_HEIGHT 512
#define RENDER_FPS 60
//#define RENDER_WITH_RAYLIB
uint8_t pixels[RENDER_WIDTH * RENDER_HEIGHT]; // Array to store the pixel data
    
const char upscale_screenshot_file[64] = "upscale_screenshot.png";
const char upscale_video_file[64] = "upscale_video.mp4";
const char ui_screenshot_file[64] = "ui_screenshot.png";
const char ui_video_file[64] = "ui_video.mp4";


void render_ui_screenshot(const char * file_name)
{
    TakeScreenshot(file_name);
    printf("Screenshot saved to %s\n", file_name);
}


void render_ui_video(const char * file_name, bool * record_frames, 
    size_t epoch, size_t frame)
{
  
    // Create a video writer
    //VideoWriter writer = VideoWriterCreate(file_name, RENDER_WIDTH, RENDER_HEIGHT, RENDER_FPS);
    while (record_frames) {
        
        /*
        for (int i = 0; i < RENDER_FPS * 10; i++) { // 10 seconds of video
        // Update the frame data
        for (size_t y = 0; y < RENDER_HEIGHT; ++y) {
            for (size_t x = 0; x < RENDER_WIDTH; ++x) {
                pixels[y * RENDER_WIDTH + x] = (uint8_t)((x + y + frame) % 256);
            }
        }
        // Write the frame to the video
        //VideoWriterWriteFrame(writer, pixels);
        frame++;
        */
    }
    return;
}


void render_upscale_video(NN nn, const char * file_name) 
{
    return;
}

void render_upscale_screenshot(NN nn, const char * file_name)
{

    for (size_t y = 0; y < (size_t) RENDER_HEIGHT; ++y){
        for (size_t x = 0; x < (size_t) RENDER_WIDTH; ++x){
            MAT_AT(NN_IN(nn), 0, 0) = (float) x / (RENDER_WIDTH - 1);
            MAT_AT(NN_IN(nn), 0, 1) = (float) y / (RENDER_HEIGHT - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUT(nn), 0, 0)*255.f;
            // Draw the pixel in the live preview image
            Color color_pixel = CLITERAL(Color) { pixel, pixel, pixel, 255 };
//            ImageDrawPixel(&img, x, y, color_pixel);
        }
    }
}

void nn_render(NN nn, int pos_x, int pos_y, int ui_w, int ui_h) 
{
    size_t arch_count = nn.count + 1;
    float  nn_layer_pad_h = ui_w / arch_count;

    size_t resiz_factor = ui_h > ui_w ? ui_h : ui_w;

    float NEURON_RADIUS = 10 + (ui_h/50) ;
    float CONX_WIDTH = 2 + ((ui_h)/100)/250;

    Color activate = COLOR_HIGH_ACTIVITY;
    Color deactivate = COLOR_LOW_ACTIVITY;
    // TODO: 
    //  - extract element values for hovering displays
    //  - add dynamic color for neuron activity 
    for (size_t l=0; l < arch_count; ++l) {

        int layer_pad_v1 = ui_h / nn.as[l].cols;

        for (size_t i=0; i < nn.as[l].cols; ++i) {

            int cx1 = pos_x + nn_layer_pad_h * l + nn_layer_pad_h  / 2;
            int cy1 = pos_y + layer_pad_v1 * i + layer_pad_v1 / 2;

            // neuron inter layers connection network
            if (l + 1 < arch_count)
            {
                int layer_pad_v2 = ui_h / nn.as[l+1].cols;

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
                DrawCircle(cx1, cy1, NEURON_RADIUS, UI_FG_COLOR);
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

/*
* Render cost graph
* @param chart: Chart structure containing cost data
* @param xpos: X position to start rendering the chart
* @param ypos: Y position to start rendering the chart
* @param img_w: Width of the chart image
* @param img_h: Height of the chart image   
*/
void cost_graph_render(Chart chart, int xpos, int ypos, int img_w, int img_h)
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
    
    char buffer_text_value[64];

    for (size_t i = 0; i+1 < chart.count; ++i) {

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
    float ref_x0 = xpos - 3 ; //+ (1 - (0 - min) / (max - min)) * img_w;
    float ref_y0 = ypos + (1 - (0 - min) / (max - min)) * img_h;
    //float ref_xmax = img_w - xpos ; //+ (1 - (0 - min) / (max - min)) * img_w;
   // float ref_ymax = ypos + (1 - (0 - min) / (max - min)) * img_h;
    // y axis line  
    DrawLineEx((Vector2){ref_x0, ref_y0}, (Vector2){ref_x0, ref_y0 - img_h}, img_h*0.004, color_ref_y0);
    // x axis line & cost/y = 0 reference 
    DrawLineEx((Vector2){ref_x0 - 1, ref_y0}, (Vector2){ ref_x0 + img_w, ref_y0}, img_h*0.004, color_ref_y0);
    DrawText("0", (int)(ref_x0 - 5), (int)(ref_y0), (int)(img_h*0.04), color_ref_y0);     	

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


void ascii_print_ref_data(int img_w, int img_h, Mat to, bool hide_null)
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

void ascii_print_nn_out_data(NN nn, int img_w, int img_h, bool hide_null)
{
    printf("*** 'ASCII styled': NN Post-Training Result image pixel data: ***\n");

    for (size_t y=0; y < (size_t)img_h; ++y){
        for (size_t x=0; x < (size_t)img_w ; ++x){
            MAT_AT(NN_IN(nn), 0, 0) = (float) x / (img_w - 1);
            MAT_AT(NN_IN(nn), 0, 1) = (float) y / (img_h - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUT(nn), 0, 0)*255.f;

            (hide_null && pixel == 0) ?
                printf("    ") :
                printf("%3u ", pixel);
        }
        printf("\n");
    }
}

void live_preview_render(Image img, Texture2D tex,
     NN nn, int img_w, int img_h, 
     int pos_x, int pos_y, int graph_w, int graph_h)
{
    for (size_t y = 0; y < (size_t) img_h; ++y){
        for (size_t x = 0; x < (size_t) img_w; ++x){
            MAT_AT(NN_IN(nn), 0, 0) = (float) x / (img_w - 1);
            MAT_AT(NN_IN(nn), 0, 1) = (float) y / (img_h - 1);
            nn_forward(nn);
            uint8_t pixel = MAT_AT(NN_OUT(nn), 0, 0)*255.f;
            // Draw the pixel in the live preview image
            Color color_pixel = CLITERAL(Color) { pixel, pixel, pixel, 255 };
            ImageDrawPixel(&img, x, y, color_pixel);
        }
    }
    pos_x += graph_w ;
    // Draw the live preview imag
    DrawTextureEx(tex, CLITERAL(Vector2) { pos_x, pos_y }, 0, 10, WHITE);
    UpdateTexture(tex, img.data);  
}

void live_original_img_render(Image img, Texture2D tex, Mat to, int img_w, int img_h, int pos_x, int pos_y, int graph_w, int graph_h)
{
    for (size_t y = 0; y < (size_t) img_h; ++y){
        for (size_t x = 0; x < (size_t) img_w; ++x){
            size_t idx = y * img_w + x;
            uint8_t pixel = MAT_AT(to, idx, 0)*255.f;
            // Draw the pixel in the live preview image
            Color color_pixel = CLITERAL(Color) { pixel, pixel, pixel, 255 };
            ImageDrawPixel(&img, x, y, color_pixel);
        }
    }      
    // Draw the original image
    DrawTextureEx(tex, CLITERAL(Vector2) { pos_x, pos_y }, 0, 10, WHITE);
    UpdateTexture(tex, img.data);   
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
NN display_training_gui(NN nn, NN g, Mat ti, Mat to,  float learn_rate, float diff, size_t max_epoch,  char * title)
{
        // Initialize the GUI window, config flags, title, sizing and FPS,  
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT, title);
    SetTargetFPS(UI_FPS);

    // preview image
    // while training, for performance concern, we want to avoid rendering a 'large' preview image
    // ie: too many pixels per frame, at 60fps, the preview is an additional load to the ui
    // TODO: 
    //  we use the original image size for now, (ie: 28x28, as we can easily re-compute the size from the ti/to matrix rows count),
    //  ie: original_img_size_w = sqrt(ti.rows) , this will work for any image data as long as it's not too large and has a ratio 1:1,
    //  NEXT TODO: otherwise we use a default preview image size_t, that is a original_img_size_w = sqrt(ti.rows) ;
    int original_img_size = sqrt(ti.rows) ; // sqrt(ti.rows) == sqrt(to.rows) 
    size_t preview_width = original_img_size; //UI_WINDOW_WIDTH / num_of_widgets;
    size_t preview_height = original_img_size; //  UI_WINDOW_HEIGHT / num_of_widgets; // preview_width, preview_height
    Image view_image_1 = GenImageColor(preview_width, preview_height, BLACK);
    printf("Preview image size: %zu x %zu\n", max_epoch, max_epoch);
    Texture2D view_texture_1 = LoadTextureFromImage(view_image_1);
    printf("Preview image size: %zu x %zu\n", max_epoch, max_epoch);
    // Set the texture filter to bilinear for better quality
    // Set the texture to the live preview image
    //SetTextureWrap(view_texture_1, TEXTURE_WRAP_CLAMP); 
    //    SetTextureFilter(view_texture_1, TEXTURE_FILTER_BILINEAR);
    Image view_image_2 = GenImageColor(preview_width, preview_height, BLACK);
    Texture2D view_texture_2 = LoadTextureFromImage(view_image_2);
   /*
    //Texture2D view_texture_2_target = LoadTextureFromImage(view_img_1_target);
    SetTextureFilter(view_texture_2, TEXTURE_FILTER_BILINEAR);
    SetTextureWrap(view_texture_2, TEXTURE_WRAP_CLAMP); 
    */ 
    for (size_t y=0; y <  original_img_size; ++y){
        for (size_t x=0; x <  original_img_size; ++x){
            size_t idx = y * original_img_size + x;
            uint8_t pixel = MAT_AT(to, idx, 0)*255.f;
            ImageDrawPixel(&view_image_2, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
        }
    }
               
    // TODO: add dynamic window layout setup/system
    size_t num_of_widgets = 2; // 2 widget as default: neuron network and cost trace
    
    // initialize the cost trace chart
    // This will hold the cost values during training and 
    // will be used to render the cost graph
    Chart cost_trace = {0};
    DArray command_list = {0};
    size_t epoch = 0;
    size_t iteration = 0;
    size_t number_of_samples = ti.rows; // number of samples in the training data
    size_t batch_count = 28; // default batch count
    size_t batch_size = number_of_samples / batch_count; // number_of_samples / batch_count; // default batch size
    if (number_of_samples % batch_count != 0) {
        batch_size = (number_of_samples + (batch_count - 1)) / batch_count ; // 1000 samples per batch
    } 
    //size_t iterations_per_frame = 60;
    //size_t epochs_per_frame = 60;
    bool stop_training = true;


    // GUI Render loop
    // The loop will run until the user stops it or the maximum number of iterations is reached
    while (!WindowShouldClose())
    {
        // __ user input setup  __
        {
            // ESC: stop & exit training UI
            if (IsKeyPressed(KEY_ESCAPE)) {
                CloseWindow();
                return nn;
            }
            // R: reset/restart current training session
            else if (IsKeyPressed(KEY_R)) {
                stop_training = true;
                // reset training
                nn_rand(nn, -1, 1);
                //nn_zero(g); // reset gradient s may not be needed and be overwriten directly to save computation time
                //da_append(&cost_trace, 0.f);
                cost_trace.count = 0;
                epoch = 0;
            }
            // S: render screenshot
            else if (IsKeyPressed(KEY_S)) {
         //      render_upscale_screenshot(nn, upscale_screenshot_file);
            }
            // S: render screenshot
            else if (IsKeyPressed(KEY_V)) {
          //      render_upscale_video(nn, upscale_video_file);
            }
            // SPACE: start/pause training
            else if (IsKeyPressed(KEY_SPACE)) { stop_training = !stop_training; }
            // F11: toggle fullscreen mode
            else if (IsKeyPressed(KEY_F11)) { ToggleFullscreen(); }   
            // UP/DOWN: increase/decrease learn rate
            else if (IsKeyPressed(KEY_UP) && learn_rate < 10.f) { learn_rate += 0.1f; }
            else if (IsKeyPressed(KEY_DOWN) && learn_rate > 0.0001f) { learn_rate -= 0.1f; }
            // LEFT/RIGHT: decrease/increase max iterations
            else if (IsKeyPressed(KEY_LEFT) && max_epoch > TRAINING_MIN_ITER) { max_epoch -= TRAINING_MIN_ITER; }
            else if (IsKeyPressed(KEY_RIGHT) && max_epoch < TRAINING_MAX_ITER) { max_epoch += TRAINING_MIN_ITER; }
            /*
                // M: toggle training method
                else if (IsKeyPressed(KEY_M)) {
                    if (strcmp(meth_name, "back_prop") == 0) { meth_name = "defined_gradient_difference"; }
                    else if (strcmp(meth_name, "defined_gradient_difference") == 0) { meth_name = "stochastic_grad_diff"; } 
                    else if (strcmp(meth_name, "stochastic_grad_diff") == 0) { meth_name = "adam"; }
                    else if (strcmp(meth_name, "adam") == 0) { meth_name = "rmsprop"; }
                    else { meth_name = "back_prop"; }
                }
            */
            /*
                // A: toggle activation function
            else if (IsKeyPressed(KEY_A)) {
                if (strcmp(act_name, "sigmoid") == 0) { act_name = "relu"; }
                else if (strcmp(act_name, "relu") == 0) { act_name = "tanh"; }
                else if (strcmp(act_name, "tanh") == 0) { act_name = "leaky_relu"; }
                else if (strcmp(act_name, "leaky_relu") == 0) { act_name = "softmax"; }
                else { act_name = "sigmoid"; }
            } */
        } // __ end of user input handling __        
        // __ training loop __
       
        float total_cost = 0.f; 
        // cost = nn_cost(nn, ti, to);
        if (epoch >= max_epoch ) { stop_training = true; }
        
        else if (!stop_training && epoch < max_epoch) {

            
            //Mat batch_list[batch_count][2]; // list of batches
            
         
            for (size_t batch_i = 0; batch_i < batch_count; ++batch_i) {
        // Initialize the batch input and output matrices
        // Each batch will have a size of batch_size
        // and will be filled with the corresponding data from ti and to matrices
                size_t start_idx = batch_i * batch_size;
                float cost = 0.f;
                Mat batch_ti = { 
                    .es = &MAT_AT(ti, start_idx, 0),
                    .cols = ti.cols,
                    .rows = batch_size,
                    .stride = ti.stride
                }; // batch input data
        
                Mat batch_to = {
                    .es = &MAT_AT(to, start_idx, 0),
                    .cols = to.cols, 
                    .rows = batch_size, 
                    .stride = to.stride
                }; // batch output data
        
                nn_backprop(nn, g, batch_ti, batch_to);
                nn_learn(nn, g, learn_rate);
                cost = nn_cost(nn, batch_ti, batch_to); //nn_cost(nn, ti, to);
                cost /= batch_size; // average cost over the batch
                total_cost += cost; // accumulate cost for the epoch
                //da_append(&cost_trace, cost); 
            }
            total_cost /= batch_count; // accumulate cost for the epoch
            da_append(&cost_trace, total_cost); 
            mat_shuffle_rows_sync(ti, to);
            epoch += 1;

            
            /*
            batch_list[batch_i][0] = batch_ti; // input data
            batch_list[batch_i][1] = batch_to; // output data
        
            } // __ end of batch list initialization __
            //if (nn.method == ALGO_SGD) {
                // Stochastic Gradient Descent training method
                for (size_t batch_i = 0; batch_i < batch_count; ++batch_i ) {
                    Mat batch_ti = batch_list[batch_i][0];
                    Mat batch_to = batch_list[batch_i][1];  
             //       for (size_t i = 0; i < batch_size; ++i ) {
                    nn_backprop(nn, g, batch_ti, batch_to);
                    nn_learn(nn, g, learn_rate);
                    cost += nn_cost(nn, batch_ti, batch_to); //nn_cost(nn, ti, to);
                    iteration += 1;
                }
                cost /= iteration; // average cost over the batch
                da_append(&cost_trace, cost); 
                epoch += 1;
            
                /*
            cost = nn_cost(nn, ti, to);
            if (nn.method == ALGO_BackProp) {

                // Backpropagation training method
                nn_backprop(nn, g, ti, to);
            }
            else if (nn.method == ALGO_FiniteDif) {
                
                // Finite Differentiation training method
                nn_finite_diff(nn, g, diff, ti, to);
            }
           
                nn_learn(nn, g, learn_rate);
                da_append(&cost_trace, cost); 
            }
       */
        } // __ end of training loop __

        BeginDrawing();
        ClearBackground(UI_BG_COLOR);
        int ui_w = GetRenderWidth();
        int ui_h = GetRenderHeight();
        // __ GUI layout __
        // Calculate the padding for the window and widgets
        int window_vpad = UI_WINDOW_PADDING * ui_w/100;
        int window_hpad = UI_WINDOW_PADDING * ui_h/100;

        int widget_pad = UI_WIDGET_PADDING * ui_w/ui_h * 100; //h > w ? UI_WIDGET_PADDING * w/100 : UI_WIDGET_PADDING * h/100;
        
        int b_graph_w     = (ui_w-2*window_vpad) / 4 ; //(num_of_widgets  1); //  * ( 1 + 2 * widget_pad)) ;
        
        int b_graph_h     = (ui_h-2*window_hpad) * 3/4; //   2;  3/4;
        int graph_pos_y = window_hpad; //h/2 - graph_h/2;
        int graph_pos_x = window_vpad; //0; // xpos = vpad;
        {   
            // Render the neural network
            int wid_graph_pos_y = ui_h/2 - b_graph_h/2;
            int wid_graph_pos_x = ui_w - b_graph_w * 2 ; // xpos;
            nn_render(nn, wid_graph_pos_x, wid_graph_pos_y, b_graph_w * 2, b_graph_h);
        }

        {
            Color color_white = WHITE;
            {   // Render the cost graph
                int wid_graph_h = b_graph_h;  // reduce the height of the cost graph
                int wid_graph_pos_y = ui_h/2 - wid_graph_h/2   ; // h2 - graph_h/2;
                int wid_graph_pos_x = window_vpad; // xpos;
                cost_graph_render(cost_trace, wid_graph_pos_x, wid_graph_pos_y, b_graph_w, wid_graph_h);
            }
            {   // DATA PANEL ( IMG RENDER OR DATA DISPLAY ....  )
                for (size_t y=0; y <  original_img_size; ++y){
                    for (size_t x=0; x <  original_img_size; ++x){
                        MAT_AT(NN_IN(nn), 0, 0) = (float) x / (original_img_size - 1);
                        MAT_AT(NN_IN(nn), 0, 1) = (float) y / (original_img_size - 1);
                        nn_forward(nn);
                        uint8_t pixel = MAT_AT(NN_OUT(nn), 0, 0)*255.f;
                        //img_out_pixels[y * img_out_w + x] = pixel;
                        ImageDrawPixel(&view_image_1, x, y, CLITERAL(Color) { pixel, pixel, pixel, 255 });
                    }
                }
                // Draw the texture to the screen
                // Render the preview images: original and NN output
                int wid_graph_w = b_graph_w / 4; ///3;  // reduce the height of the cost graph
                int wid_graph_h =  b_graph_h / 4; ///3;  // reduce the height of the cost graph
                int wid_graph_pos_y = ui_h/2 - wid_graph_h/2; // h2 - graph_h/2;
                int wid_graph_pos_x = window_vpad +  b_graph_w + wid_graph_w; // xpos;
                int scale = 5; // scale factor for the preview images
                
        
             //   graph_pos_y = graph_pos_y + graph_h; //h2 - graph_h/2;
               // graph_pos_x = window_vpad; // xpos;
                //graph_h = graph_h / 3; // graph_h + 2*widget_pad; // preview_width;
               // graph_w = graph_h; // preview_width;
            //live_preview_render(view_image_1, view_texture_1, nn, original_img_size, original_img_size, graph_pos_x, graph_pos_y, graph_w, graph_h);
                        // Uaapdate the texture with the new image data
                UpdateTexture(view_texture_1, view_image_1.data);
            // Draw the texture to the screen
                DrawTextureEx(view_texture_1, CLITERAL(Vector2) { wid_graph_pos_x, wid_graph_pos_y }, 0, scale, color_white);
            // Render the second preview image
           // graph_pos_x += graph_w + widget_pad; // xpos + graph_w + widget_pad
            // Render the second preview image
        //   DrawTextureEx(view_texture_2, CLITERAL(Vector2) { wid_graph_pos_x, wid_graph_pos_y }, 0, 10, WHITE);
           
            //live_preview_render(view_image_2, view_texture_2, nn, graph_w, graph_h, graph_pos_x + graph_w, graph_pos_y, graph_w, graph_h);
          
                UpdateTexture(view_texture_2, view_image_2.data);
            // Draw the texture to the screen
                DrawTextureEx(view_texture_2, CLITERAL(Vector2) { wid_graph_pos_x, wid_graph_pos_y - wid_graph_h }, 0, scale, color_white);
       
            }
            {
                // UI header text
                int header_x = window_vpad;
                int header_y = window_hpad;
                int header_w = ui_w - 2 * window_vpad;
                int header_h = (ui_h  - 2 * window_vpad) * 0.08; // 8% of the window
                DrawRectangleLines(header_x, header_y, header_w, header_h, UI_FG_COLOR); 
                // Draw the title text
                char buffer[256];
                snprintf(buffer, sizeof(buffer),
                   "Epoch: %zu/%zu - Rate: %f - Cost: %f - Algo: %d & Act: %d", 
                   epoch, max_epoch, learn_rate, total_cost, nn.method, nn.activation);
                DrawText(buffer, header_x + ui_h*0.02, header_y + ui_h*0.02, ui_h*0.04, color_white);
            }
             if (epoch == 0 || stop_training) {
                DrawText("Press SPACE to train...", 150, 150, ui_h*0.04, color_white); //     
            }
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
    printf("Upscaled image size: %d x %d\n", out_rescaled_img_size_w, out_rescaled_img_size_h);
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

    // Load the training data from the specified data file
    Mat t = mat_load(data_file_name);

    ML_ASSERT(arch.count > 1);
    ML_ASSERT(t.rows > 0 && t.cols > 0);
    

    size_t arch_in_sz = arch.items[0] ;
    size_t arch_out_sz = arch.items[arch.count - 1];
    ML_ASSERT(t.cols == arch_in_sz + arch_out_sz);

   // Set the input and output sizes based on the architecture
    // Split the Loaded Trainining data Matrix into input and output matrices
    // ti: input matrix, to: output matrix
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
    // Allocate a neural network with the specified architecture
    NN nn = nn_alloc(arch.items, arch.count);    
    // Allocate a duplicate neural network for backpropagation
    // This will hold the gradients during training
    // This is used to update the weights and biases during training
    NN g = nn_alloc(arch.items, arch.count);
    // Initialize the neural network with random weights and biases
    nn_rand(nn, -1, 1);
    
    // TODO: make the title dynamic
    char * title = "ML_LIB.C -- ML_Viewer -- NN Training GUI";
    char * ui_title = "Image Upscaling Image with Neural Network";
    
    
    // Set the training method and parameters
    // TODO: use enum for training method and activation function
    //nn.method = ALGO_SGD; // Set the training method
    //nn.activation = Activ_Sig; // Set the activation function

    size_t EPOCHS = TRAINING_DEFAULT_ITER;
    // TO DO: make the learning rate a dynamic function that automatically adjusts
    // For now, we will use a fixed learning rate 
    float_t learn_rate = TRAINING_DEFAULT_LEARN_RATE; //1.f; //0.1f; 
    float_t diff_delta = TRAINING_DEFAULT_DIFF_DELTA; //1.f; //1e-1;
    
    
    printf("Training data: %zu rows, %zu cols\n", t.rows, t.cols);

    nn = display_training_gui(nn, g, ti, to, learn_rate, diff_delta, EPOCHS, title);
    
    size_t original_img_size_h = sqrt(t.rows) ;
    size_t original_img_size_w = original_img_size_h;
    //TODO: HANDLE ASPECT RATIO for non-square images
    // TO DO: handle non-square images WITH ASPECT RATIO FOR UPSCALING
    
    if (original_img_size_w * original_img_size_h != t.rows) {
        fprintf(stderr, "Ratio error: not supported, image size is not square: %zu x %zu\n", original_img_size_w, original_img_size_h);
        return 1;
    }
    
    ascii_print_ref_data(original_img_size_w, original_img_size_h, to, false);
    ascii_print_nn_out_data(nn, original_img_size_w, original_img_size_h, false);
    
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
 
    // save upscaled image
    if (!stbi_write_png(out_rescaled_file_name, img_out_w, img_out_h, 1, img_out_pixels, 0)) 
    {
        fprintf(stderr, "Error: failed to save upscaled image: %s\n", out_rescaled_file_name);
        return 1;
    }
   
    printf("upscaled image %zu x %zu saved as %s\n", img_out_w, img_out_h, out_rescaled_file_name); 
    free(img_out_pixels);
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

