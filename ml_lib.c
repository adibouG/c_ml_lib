#define ML_IMP
#include "ml_lib.h"

//typedef float sample[];
 
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
    float delta = 1e-1;
    float learn_rate = 1e-1;
    
    size_t ITER = 2000;
    
    for (size_t i=0;i<ITER;++i){
        nn_finite_diff(nn, g, delta, ti, to);
        nn_learn(nn, g, learn_rate);
        printf("%zu: cost=%f\n", i, nn_cost(nn, ti, to)); 
    }
    
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
