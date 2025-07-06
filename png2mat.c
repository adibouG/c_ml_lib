// gcc -o png2mat png2mat.c -lm 
// gcc -o png2mat png2mat.c -lm -lpng -lz -I./stb_image
#define ML_IMP
#include "mllib.h"

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>

//#include "raylib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ext_libs/stb/stb_image.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "missing input argument 1: <image filename>\n");
        return 1;
    }
    char * filename = argv[1];
    char * dataname = "imgpixel.mat";
    if (argc == 3) {
        dataname = argv[2];
    } else {
        printf("missing argument 2 specifying a path or name for output file.\n...using default: %s\n", dataname);
    }
    printf("output filename: %s\n", dataname);
    
    int img_w, img_h, img_c;
    uint8_t * pixel_data = (uint8_t *) stbi_load(filename, &img_w, &img_h, &img_c, 0);
    
    if (pixel_data == NULL) {
        fprintf(stderr, "failed to load image file: %s\n", filename);
        return 1;
    }
    if (img_c != 1 && img_c != 3) {
        fprintf(stderr, "image must be grayscale or RGB, but has %d channels\n", img_c);
       // stbi_image_free(pixel_data);
        return 1;
    }

    printf("filename: %s:\n\
        - size: %d x %d (%f\n)\n\
        - bits/pixel: %d (%dx8)\n",
         filename, img_w, img_h, img_w/(float)img_h, img_c*8, img_c );
    // row-major matrix with 3 columns: x, y, pixel_value
    Mat m = mat_alloc(img_w*img_h, 3) ;
    
    for (size_t y=0; y < (size_t) img_h; ++y){
        for (size_t x=0; x < (size_t) img_w; ++x){
            size_t i = (y * img_w + x ) ;
            MAT_AT(m, i , 0) = (float) x/(img_w -1) ;
            MAT_AT(m, i , 1) = (float) y/(img_h -1) ;
            MAT_AT(m, i , 2) = pixel_data[i]/255.f ;
            printf("%3u ", pixel_data[i]);
        }
        printf("\n");
    }
    
    mat_save(dataname, m);
    printf("image pixels extracted as float matrix and saved as .mat file\n\
            - input image file: %s\n\
            - output file: %s\n", filename, dataname);
    
    return 0;
}
