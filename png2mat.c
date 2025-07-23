// gcc -o png2mat png2mat.c -lm 
// gcc -o png2mat png2mat.c -lm -lpng -lz -I./stb_image
#define ML_IMP
#include "mllib.h"

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>     
//#include "raylib.h"

#define STB_IMAGE_IMPLEMENTATION
#include "ext_libs/stb/stb_image.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "missing input argument 1: <image filename>\n");
        return 1;
    }
    char * dataout_name = "imgpixel.mat";
    char * filename1 = argv[1];
    char * filename2 = argc > 2 ? argv[2] : NULL; //argv[2]; // "imgpixel2.mat";
    if (argc > 3) {
        dataout_name = argv[3];
    } else {
        printf("missing output argument , using default name:\n");
    }
    printf("output filename: %s\n", dataout_name);
    size_t * pixel_data = NULL; 
    size_t image_count = 0;

    enum { FALSE = 0, TRUE } try_file = FALSE;
    enum { NONE, GRAYSCALE, RGB, RGBA, NOT_SUPPORTED } image_type = NONE;

    if (filename1 != NULL) {
        printf("input filename 1: %s\n", filename1);
        if (access(filename1, F_OK) == -1) {
            fprintf(stderr, "file %s does not exist\n", filename1);
        }
        else if (access(filename1, R_OK) == -1) {
            fprintf(stderr, "file %s is not readable\n", filename1);
        }   
        else {
            try_file = TRUE;
        }
    }
    
    uint32_t max_value = UINT32_MAX ;// Use UINT32_MAX to represent the maximum value for a 32-bit unsigned integer
    uint32_t img1_w, img1_h, img1_c;
   // uint32_t img1_c = 0x0; // channels
    uint32_t * pixel_data1 = NULL;
    if (try_file){
       
        pixel_data1 = (uint32_t *) stbi_load(filename1, &img1_w, &img1_h, &img1_c, 4); 
        
        if (pixel_data1 == NULL) {
            fprintf(stderr, "failed to load image file: %s\n", filename1);    
        }
        else {
            printf("loaded image file: %s\n", filename1);
            image_count = 1;
        }
        if (img1_c != 1 && img1_c != 3 && img1_c != 4) {
            fprintf(stderr, "image must be grayscale *or* RGB or RGBA, but instead has %d channels\n", (int) img1_c);
           // stbi_image_free(pixel_data);
        } 
        try_file = 0; 
    }
    
    if (filename2 != NULL) {
        printf("input filename 2: %s\n", filename2);
        if (access(filename2, F_OK) == -1) {
            fprintf(stderr, "file %s does not exist\n", filename2);
        }
        else if (access(filename2, R_OK) == -1) {
            fprintf(stderr, "file %s is not readable\n", filename2);
        }   
        else {
            try_file = 1;
        }
    }

    int img2_w, img2_h, img2_c;
    uint32_t * pixel_data2 = NULL;
    
    if (try_file) {

        pixel_data2 = (size_t *) stbi_load(filename2, &img2_w, &img2_h, &img2_c, 0);
    
        if (pixel_data2 == NULL) {
            fprintf(stderr, "failed to load image file: %s\n", filename2);
        }
        else {
            printf("loaded image file: %s\n", filename2);
            image_count = 2;
        }
        if (img2_c != 1 && img2_c != 3 && img2_c != 4) {
            fprintf(stderr, "image must be grayscale or RGB, but has %d channels\n", img2_c);
        }
        try_file = 0;
    }


    printf("filename: %s:\n\
        - size: %d x %d (%f\n)\n\
        - bits/pixel: %d (%dx8)\n",
         filename1, img1_w, img1_h, img1_w/(float)img1_h, img1_c*8, img1_c );
    
    printf("filename: %s:\n\
        - size: %d x %d (%f\n)\n\
        - bits/pixel: %d (%dx8)\n",
         filename2, img2_w, img2_h, img2_w/(float)img2_h, img2_c*8, img2_c );
    // row-major matrix with 4 columns: x, y, INDEX, pixel_value
    Mat m = mat_alloc(img1_w*img1_h + img2_w*img2_h, 4) ;
    
    for (size_t y=0; y < (size_t) img1_h; ++y){
        for (size_t x=0; x < (size_t) img1_w; ++x){
            size_t i = (y * img1_w + x ) ;
            MAT_AT(m, i , 0) = (float) x/(img1_w -1) ;
            MAT_AT(m, i , 1) = (float) y/(img1_h -1) ;
            MAT_AT(m, i , 2) = 0.f; // index 0 for first image;
            uint32_t img_c = pixel_data1[i] % 0xFFFFFFFF ;// pixel_data2[idx]/255.f ;(int8_t) img1_c;
            uint8_t img_a = ( img_c >> 24) & 0xFF;
            uint8_t img_r = ( img_c >> 16) & 0xFF;
            uint8_t img_g = ( img_c >> 8) & 0xFF;
            uint8_t img_b = ( img_c  & 0xFF );

            printf("img_c: %x, img_a: %d, img_r: %d, img_g: %d, img_b: %d\n", img_c, img_a, img_r, img_g, img_b);
            
            uint32_t img32 = (
                (img_a << 24) | (img_r << 16) | (img_g << 8) | img_b
            );

            MAT_AT(m, i , 3) = (float ) img32; //[i]/(255*img1_c) ;// pixel_data2[idx]/255.f ;
            printf("%u - ", pixel_data1[i]);
            printf(" %x : %u :  ",img32, pixel_data1[i] );
        }
        printf("\n");
    }
    
    printf("\n");
    printf("\n");
    printf("\n");

    for (size_t y=0; y < (size_t) img2_h; ++y){
        for (size_t x=0; x < (size_t) img2_w; ++x){
            size_t idx = (y * img2_w + x ); 
            size_t i = (img1_w*img1_h) + idx; // offset by img1_w*img1_h) 
            MAT_AT(m, i , 0) = (float) x/(img2_w -1) ;
            MAT_AT(m, i , 1) = (float) y/(img2_h -1) ;
            MAT_AT(m, i , 2) = 1.f; // index 0 for first image;
            MAT_AT(m, i , 3) = pixel_data2[idx]/(255.f*img2_c) ; // pixel_data2[idx]/256.f ;
            printf("%9f ", (float) pixel_data2[idx]);
        }
        printf("\n");
    }
    

    mat_save(dataout_name, m);
    printf("image pixels extracted as float matrix and saved as .mat file\n\
            - input image file1: %s\n\
            - input image file2: %s\n\
            - output file: %s\n", filename1, filename2, dataout_name);
    
    return 0;
}
