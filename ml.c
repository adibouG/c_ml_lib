#include "ml.h"

#ifndef Mat
struct Mat {
	size_t rows;
	size_t cols;
    size_t stride;
	float *es;
};
#endif

typedef struct {
    
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // amount activations count +1 
        
} NN;

#define ARRAY_LEN(xs) sizeof(xs)/sizeof((xs)[0])
// arch[] = {2,2,1}
NN nn_alloc(size_t * arch, size_t arch_count)
{
    ML_ASSERT(arch_count > 1);
    NN nn;
    nn.count = arch_count - 1;
    nn.ws = ML_MALLOC(sizeof(*nn.ws)*nn.count);
    ML_ASSERT(nn.ws != NULL);
    nn.bs = ML_MALLOC(sizeof(*nn.bs)*nn.count);
    ML_ASSERT(nn.bs != NULL);
    nn.as = ML_MALLOC(sizeof(*nn.as)*(nn.count + 1));
    ML_ASSERT(nn.as != NULL);

    nn.as[0] = mat_alloc(1, arch[0]);
    for (size_t i=1; i < arch_count; ++i){
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        nn.as[i] = mat_alloc(1, arch[i]);
    }
    return nn;    
}



typedef float sample[3];
sample and_td[] = {
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1},
};
sample or_td[] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1},
};
sample xor_td[] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};
sample * train = xor_td;
size_t  td_count = 4;

float td[] = {
    0,0,0,
    0,1,0,
    1,0,0,
    1,1,1,
};

Xor xor_rand(void){
    Xor m;
    m.w1or = rand_float();
    m.w2or = rand_float();
    m.bor = rand_float();
    m.w1nand = rand_float();
    m.w2nand = rand_float();
    m.bnand = rand_float();
    m.w1and = rand_float();
    m.w2and = rand_float();
    m.band = rand_float();
    return m;
}


float xor_forward(Xor m, float x1, float x2)
{
    float a = sigmf( m.w1or * x1 +  m.w2or * x2 +  m.bor);
    float b = sigmf( m.w1nand * x1 +  m.w2nand *x2 +  m.bnand);
    return sigmf ( m.w1and * a + m.w2and * b + m.band); 
    /*
    mat_dot(m.a1, m.in, m.w1);
    mat_sum(m.a1, m.b1); 
    mat_sig(m.a1);
    
    mat_dot(m.a2, m.in, m.w2);
    mat_sum(m.a2, m.b2); 
    mat_sig(m.a2); 
    */
}

float xor_cost (Xor m)// , Mat ti, Mat to)
{
 //   assert(ti.rows == to.rows);
  //  assert(to.cols == m.a2.cols);
    
    //size_t ti_r = ti.rows;
    
    float cost = 0.0f;
    
    for (size_t i=0; i < td_count; ++i){
        /*Mat x = mat_getrow(ti, i);
        Mat out_expect = mat_getrow(to, i);
        mat_copy(m.in, x);
        */
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = xor_forward(m, x1, x2);
        float diff = y -train[i][2];
      /*  size_t to_c = to.cols;
        for (size_t j=0; j < to_c; ++j){
            float diff = MAT_AT(m.a2, 0, j) - MAT_AT(out_expect, 0, j);
        */
        cost += diff * diff;
    //    } 
    }
            
    return cost / td_count; //ti_r;
}

Xor xor_finite_diff(Xor m, Xor g, float delta) //, Mat d_in, Mat d_out)
{
    float saved = 0;
    float cost_orig = xor_cost(m);//, d_in, d_out);
        
    saved =  m.w1or;
    m.w1or += delta;
    g.w1or = (xor_cost(m) - cost_orig)/delta;
    m.w1or = saved;
    
    saved =  m.w2or;
    m.w2or += delta;
    g.w2or = (xor_cost(m) - cost_orig)/delta;
    m.w2or = saved;
    
    saved =  m.bor;
    m.bor += delta;
    g.bor = (xor_cost(m) - cost_orig)/delta;
    m.bor = saved;
    
    saved =  m.w1nand;
    m.w1nand += delta;
    g.w1nand = (xor_cost(m) - cost_orig)/delta;
    m.w1nand = saved;
    
    saved =  m.w2nand;
    m.w2nand += delta;
    g.w2nand = (xor_cost(m) - cost_orig)/delta;
    m.w2nand = saved;
        
    saved =  m.bnand;
    m.bnand += delta;
    g.bnand = (xor_cost(m) - cost_orig)/delta;
    m.bnand = saved;
    
    saved =  m.w1and;
    m.w1and += delta;
    g.w1and = (xor_cost(m) - cost_orig)/delta;
    m.w1and = saved;
    
    saved =  m.w2and;
    m.w2and += delta;
    g.w2and = (xor_cost(m) - cost_orig)/delta;
    m.w2and = saved;
        
    saved =  m.band;
    m.band += delta;
    g.band = (xor_cost(m) - cost_orig)/delta;
    m.band = saved;
    
    return g;
    /*
    for (size_t row_x=0; row_x< m.w1.rows; ++row_x){
        for (size_t col_x=0; col_x < m.w1.cols; ++col_x){
            saved = MAT_AT(m.w1, row_x, col_x);
            MAT_AT(m.w1, row_x, col_x) += delta;
            MAT_AT(g.w1, row_x, col_x) = (xor_cost(m, d_in, d_out) - cost_orig) / delta;
            MAT_AT(m.w1, row_x, col_x) = saved;
         }
    }
    
    for (size_t row_x=0; row_x< m.w2.rows; ++row_x){
        for (size_t col_x=0; col_x < m.w2.cols; ++col_x){
            saved = MAT_AT(m.w2, row_x, col_x);
            MAT_AT(m.w2, row_x, col_x) += delta;
            MAT_AT(g.w2, row_x, col_x) = (xor_cost(m, d_in, d_out) - cost_orig) / delta;
            MAT_AT(m.w2, row_x, col_x) = saved;
        }
    }   
    
    
    for (size_t row_x=0; row_x< m.b1.rows; ++row_x){
        for (size_t col_x=0; col_x < m.b1.cols; ++col_x){
            saved = MAT_AT(m.b1, row_x, col_x);
            MAT_AT(m.b1, row_x, col_x) += delta;
            MAT_AT(g.b1, row_x, col_x) = (xor_cost(m, d_in, d_out) - cost_orig) / delta;
            MAT_AT(m.b1, row_x, col_x) = saved;
        }
    }   
    
    for (size_t row_x=0; row_x< m.b2.rows; ++row_x){
        for (size_t col_x=0; col_x < m.b2.cols; ++col_x){
            saved = MAT_AT(m.b2, row_x, col_x);
            MAT_AT(m.b2, row_x, col_x) += delta;
            MAT_AT(g.b2, row_x, col_x) = (xor_cost(m, d_in, d_out) - cost_orig) / delta;
            MAT_AT(m.b2, row_x, col_x) = saved;
        }
    } */     
}

Xor xor_learn(Xor m, Xor g, float learn_rate)
{
    
    m.w1or -= g.w1or * learn_rate;
    m.w2or -= g.w2or * learn_rate;
    m.bor -= g.bor * learn_rate;

    m.w1nand -= g.w1nand * learn_rate;
    m.w2nand -= g.w2nand * learn_rate;
    m.bnand -= g.bnand * learn_rate;
    
    m.w1and -= g.w1and * learn_rate;
    m.w2and -= g.w2and * learn_rate;
    m.band -=  g.band * learn_rate;
    return m;
    /*
    for (size_t row_x=0; row_x< m.w1.rows; ++row_x){
        for (size_t col_x=0; col_x < m.w1.cols; ++col_x){
            MAT_AT(m.w1, row_x, col_x) -= learn_rate * MAT_AT(g.w1, row_x, col_x);
        }
    }      
    
    
    for (size_t row_x=0; row_x< m.b1.rows; ++row_x){
        for (size_t col_x=0; col_x < m.b1.cols; ++col_x){
            MAT_AT(m.b1, row_x, col_x) -= learn_rate * MAT_AT(g.b1, row_x, col_x);
        }
    }      
    
    for (size_t row_x=0; row_x< m.w2.rows; ++row_x){
        for (size_t col_x=0; col_x < m.w2.cols; ++col_x){
            MAT_AT(m.w2, row_x, col_x) -= learn_rate * MAT_AT(g.w2, row_x, col_x);
        }
    }      
    
    for (size_t row_x=0; row_x< m.b2.rows; ++row_x){
        for (size_t col_x=0; col_x < m.b2.cols; ++col_x){
            MAT_AT(m.b2, row_x, col_x) -= learn_rate * MAT_AT(g.b2, row_x, col_x);
        }
    } */     
}


int main()
{
    srand(time(0));
    
   size_t stride = 3;
   size_t len = (sizeof(td)/sizeof(td[0]))/stride;

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

    MAT_PRINT(ti);

    MAT_PRINT(to);*/
    Xor m = xor_rand();
    Xor g ;    /*
    Xor m = xor_alloc();
    Xor g = xor_alloc();
    
    mat_rand(m.b2 ,0, 1); 
    mat_rand(m.w1, 0, 1);
    mat_rand(m.b1 ,0, 1);   
    mat_rand(m.w2 ,0, 1);

    */
    float delta = 1e-1;
    float learn_rate = 1e-1;    
   
   for (size_t iter=0; iter < 50000; ++iter){
        g = xor_finite_diff(m, g, delta) ; //, ti, to);
        printf("finite diff cost = %f\n", xor_cost(m));// ti, to)); 
        m = xor_learn(m, g, learn_rate); 
        printf("cost = %f\n", xor_cost(m)); //, ti, to)); 
   }
   printf("***********************\n"); 

   printf("Post training Model Test\n"); 
            
    for (size_t i=0; i < 2; ++i){
        for (size_t j=0; j < 2; ++j){
            printf("%zu ^ %zu = %f\n", i, j, xor_forward(m,i,j)); 
      } 
    }

    return 0;
}
