#ifndef ML_H_
#define ML_H_

#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef ML_MALLOC
#define ML_MALLOC malloc
#endif

#ifndef ML_ASSERT
#include <assert.h>
#define ML_ASSERT assert
#endif

typedef enum {
    Sig = 0,
    Rel,
    Sin,
} Activation;


typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float_t *es;
} Mat;




float rand_float(void);
//activation fn
float sigmf(float x);
float reluf(float x);
//float sinf(float x);
//matrice fn
Mat mat_alloc(size_t rows, size_t cols);
Mat mat_getrow(Mat m, size_t row);
Mat mat_getsubmat(Mat m, size_t from_row, size_t to_row, size_t from_col, size_t to_col);

Mat mat_sub(Mat m, size_t start);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_rand(Mat m, float min, float max);
void mat_print(Mat m, const char * layer, size_t padding);
void mat_fill(Mat m, float val);

void mat_sin(Mat m);
void mat_sig(Mat m);
void mat_relu(Mat m);

void mat_activate_fn(Mat m, Activation f);

#define MAT_PRINT(m) mat_print(m, #m, 0);
#define MAT_AT(m, r, c) (m).es[(r)*(m).stride + (c)]


typedef struct { 
    float r, g, b, a; 
} PixelRGBA;

int32_t pixel_rgba_to_int(PixelRGBA p)
{
    return ((int32_t)(p.r * 255) << 24) | ((int32_t)(p.g * 255) << 16) |
           ((int32_t)(p.b * 255) << 8) | (int32_t)(p.a * 255);
} 

PixelRGBA pixel_int_to_rgba(int64_t p)
{
     (PixelRGBA){
        .r = ((p >> 24) & 0xFF) / 255.f,
        .g = ((p >> 16) & 0xFF) / 255.f,
        .b = ((p >> 8) & 0xFF) / 255.f,
        .a = (p & 0xFF) / 255.f,
    };
}

#define PRINT_AS_HEX(p) \
    printf("0x%08x\n", p)

typedef struct {
    size_t width;
    size_t height;
    PixelRGBA *data;
} ImageRGBA;



typedef struct {
    Mat * mat;
    Activation a;   
} NNLayer;


typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // amount activations count +1 
} NN;


void nn_print (NN nn, const char * layer);
void nn_rand(NN m, float min, float max);
NN nn_alloc(size_t * arch, size_t arch_count);

void nn_zero(NN n);
float nn_cost (NN nn, Mat t_in, Mat t_out);
void nn_forward(NN nn);
void nn_finite_diff(NN nn, NN g, float delta, Mat t_in, Mat t_out);
void nn_backprop(NN nn, NN g, Mat t_in, Mat t_out);
void nn_learn(NN m, NN g, float learn_rate);

#define NN_PRINT(nn) nn_print(nn, #nn)
#define ARRAY_LEN(xs) sizeof(xs)/sizeof((xs)[0])
#define NN_IN(nn) (nn).as[0]
#define NN_OUT(nn) (nn).as[(nn).count]

#endif

#ifdef ML_IMP

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

float sigmf(float x){
    return 1.f / (1.f + expf(-x));
}

void mat_sig(Mat a)
{
    for (size_t r=0; r< a.rows; ++r){
        for (size_t c=0; c< a.cols; ++c){
             MAT_AT(a, r, c) = sigmf(MAT_AT(a, r, c));
        }       
    }
}   

float reluf(float x){
    return x < 0 ? 0 : x;
}

void mat_relu(Mat a)
{
    for (size_t r=0; r< a.rows; ++r){
        for (size_t c=0; c< a.cols; ++c){
             MAT_AT(a, r, c) = reluf(MAT_AT(a, r, c));
        }
    }
}   

void mat_sin(Mat a)
{
    for (size_t r=0; r< a.rows; ++r){
        for (size_t c=0; c< a.cols; ++c){
             MAT_AT(a, r, c) = sinf(MAT_AT(a, r, c));
        }
        
    }
}   

void mat_activate_fn(Mat a, Activation f)
{
    for (size_t r=0; r< a.rows; ++r){
        for (size_t c=0; c< a.cols; ++c){
            switch (f){
                case Sig: 
                    MAT_AT(a, r, c) = sigmf(MAT_AT(a, r, c));
                    break;
                case Rel: 
                    MAT_AT(a, r, c) = reluf(MAT_AT(a, r, c));
                    break;
                case Sin: 
                    MAT_AT(a, r, c) = sinf(MAT_AT(a, r, c));
                    break;
                default: 
                    MAT_AT(a, r, c) = sigmf(MAT_AT(a, r, c));
                    break;
            }                
        }
        
    }
}   


Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;  
    m.rows =rows;
    m.cols = cols;
    m.stride = cols;
    m.es = ML_MALLOC(sizeof(*m.es)*rows*cols);
    
    ML_ASSERT(m.es != NULL);
    return m;
}

Mat mat_getrow(Mat m, size_t row)
{
    ML_ASSERT(row < m.rows);
    
    return (Mat){
        .rows   = 1,
        .cols   = m.cols,       
        .stride = m.stride,
        .es     = &MAT_AT(m, row, 0),
    };
}   

Mat mat_getsubmat(Mat m, size_t fromrow, size_t torow, size_t fromcol, size_t tocol)
{
    ML_ASSERT(torow < m.rows);
    ML_ASSERT(tocol < m.cols);
    
    size_t nrows = torow - fromrow;
    size_t ncols = tocol - fromcol;
    
    Mat e = mat_alloc(nrows, ncols);
    
    for (size_t r=0; r < nrows; ++r){
        for (size_t c=0; c < ncols; ++c){
            MAT_AT(e, r, c) = MAT_AT(m, fromrow + r, fromcol + c);
        }
    }
    return e; 
}

Mat mat_sub(Mat m, size_t start)
{
    /*
     * float d[] = {
     *  0,0,0,
     *  0,1,1,
     *  1,0,1,
     *  1,1,1
     * };
     *  Mat sub_i = {rows=4, cols=2,stride=3, es=&d[0] or d}
     *  Mat sub_o = {rows=4, cols=1,stride=3, es=&d[2] or d + 2}
     */
     return (Mat){
        .rows   = m.rows,
        .cols   = m.cols - start ,      
        .stride = m.stride,
        .es     = m.es + start,
    };
}   
    

void mat_copy(Mat dst, Mat src)
{
    ML_ASSERT(dst.rows == src.rows && dst.cols == src.cols);
    
    for (size_t r=0; r < src.rows; ++r){
        for (size_t c=0; c < src.cols; ++c){
            MAT_AT(dst, r, c) = MAT_AT(src, r, c);
        }
    }
}

    
    
void mat_dot(Mat dst, Mat a, Mat b)
{
    ML_ASSERT(b.rows==a.cols);
    ML_ASSERT(dst.rows==a.rows && dst.cols==b.cols);
    
    size_t o = a.cols;
    for (size_t r=0; r< dst.rows; ++r){
        for (size_t c=0; c< dst.cols; ++c){
            MAT_AT(dst, r, c) = 0;
            for (size_t k=0; k< o; ++k){
                MAT_AT(dst, r, c) += MAT_AT(a, r, k) * MAT_AT(b, k, c);
            }
        }
    }
}



void mat_sum(Mat dst, Mat a)
{
    ML_ASSERT(dst.rows==a.rows && dst.cols==a.cols);
    for (size_t r=0; r< dst.rows; ++r){
        for (size_t c=0; c< dst.cols; ++c){
             MAT_AT(dst, r, c) += MAT_AT(a, r, c);
        }
        
    }
}   


void mat_rand(Mat m, float min, float max)
{
    for (size_t r=0; r< m.rows; ++r){
        for (size_t c=0; c< m.cols; ++c){
             MAT_AT(m, r, c) = rand_float()*(max - min) + min ;
        }       
    }
}


void mat_fill(Mat m, float val)
{
    for (size_t r=0; r< m.rows; ++r){
        for (size_t c=0; c< m.cols; ++c){
             MAT_AT(m, r, c) = val ;
        }
    }
}


void mat_print(Mat m, const char * layer, size_t padding)
{
    printf("%*s%s : [\n", (int) padding, "", layer);
    for (size_t r=0; r < m.rows; ++r){
    printf("%*s    " , (int) padding, "");
    for (size_t c=0; c < m.cols; ++c){
        printf("%f ", MAT_AT(m, r, c));
    }
    printf("\n");
    }
    printf("%*s]\n" , (int) padding, "");
}


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


void nn_rand(NN m, float min, float max)
{
    for (size_t i=0; i< m.count; ++i){
        mat_rand(m.ws[i], min, max);
        mat_rand(m.bs[i], min, max);
    }
}

void nn_forward(NN nn)
{
    for (size_t i=0; i < nn.count; ++i){
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);   
        mat_sum(nn.as[i+1] , nn.bs[i]);   
        mat_sig(nn.as[i+1]);
    }
}


float nn_cost (NN nn, Mat t_in, Mat t_out)
{
    assert(t_in.rows == t_out.rows);
    assert(t_out.cols == NN_OUT(nn).cols);
    
    size_t n = t_in.rows;
    
    float cost = 0;
    
    for (size_t i=0; i < n; ++i){
        Mat x = mat_getrow(t_in, i);
        Mat y = mat_getrow(t_out, i); //expected output result for input x (aka f(x) = y)
        mat_copy(NN_IN(nn), x);
        nn_forward(nn);
        size_t m = t_out.cols;
        for (size_t j=0; j < m; ++j){
            float d = MAT_AT(NN_OUT(nn), 0, j) - MAT_AT(y, 0, j);
            cost += d * d;
        } 
    }
            
    return cost / n;
}

void nn_finite_diff(NN m, NN g, float delta, Mat t_in, Mat t_out)
{
    float saved;
    float cost_orig = nn_cost(m, t_in, t_out);
    
    for (size_t i=0; i< m.count; ++i){
        for (size_t row_x=0; row_x< m.ws[i].rows; ++row_x){
            for (size_t col_x=0; col_x < m.ws[i].cols; ++col_x){
                saved = MAT_AT(m.ws[i], row_x, col_x);
                MAT_AT(m.ws[i], row_x, col_x) += delta;
                MAT_AT(g.ws[i], row_x, col_x) = (nn_cost(m, t_in, t_out) - cost_orig) / delta;
                MAT_AT(m.ws[i], row_x, col_x) = saved;
            }
        }
    
        for (size_t row_x=0; row_x< m.bs[i].rows; ++row_x){
            for (size_t col_x=0; col_x < m.bs[i].cols; ++col_x){
                saved = MAT_AT(m.bs[i], row_x, col_x);
                MAT_AT(m.bs[i], row_x, col_x) += delta;
                MAT_AT(g.bs[i], row_x, col_x) = (nn_cost(m, t_in, t_out) - cost_orig) / delta;
                MAT_AT(m.bs[i], row_x, col_x) = saved;  
            }
        }   
    }
}

void nn_zero(NN n)
{
    for (size_t i=0; i < n.count; ++i){
            mat_fill(n.ws[i], 0);
            mat_fill(n.bs[i], 0);
            mat_fill(n.as[i], 0);
    }
    mat_fill(n.as[n.count], 0);
}    
        

void nn_backprop(NN nn, NN g, Mat t_in, Mat t_out)
{
    ML_ASSERT(t_in.rows == t_out.rows);
    ML_ASSERT(NN_OUT(nn).cols == t_out.cols);


    size_t n = t_in.rows;

    nn_zero(g);
    
    // i: current input sample
    // l: current layer
    // j: current activation 
    // k:  previous activation
     
    for (size_t i=0; i< n; ++i){
        mat_copy(NN_IN(nn), mat_getrow(t_in, i));
        nn_forward(nn);
    
    
        for (size_t j=0; j <= nn.count; ++j){
            mat_fill(g.as[j], 0);
    }
        for (size_t j=0; j < t_out.cols; ++j){
            MAT_AT(NN_OUT(g), 0, j) = MAT_AT(NN_OUT(nn), 0, j) - MAT_AT(t_out, i, j);
        }
        
        for (size_t l= nn.count; l > 0; --l){
            for (size_t j=0; j < nn.as[l].cols; ++j){
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                
                MAT_AT(g.bs[l-1], 0, j) += 2 * da*a*(1-a);
                
                for (size_t k=0; k < nn.as[l-1].cols; ++k){
                    // j - weight matrix col 
                    // k - weight matrix row 
                
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += 2 * da*a*(1-a)*pa;
                    MAT_AT(g.as[l-1], 0, k) += 2 * da*a*(1-a)*w;
                }
            }
        }
    }
    for (size_t i=0; i< g.count; ++i){
        for (size_t j=0; j< g.ws[i].rows; ++j){
            for (size_t k=0; k< g.ws[i].cols; ++k){
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j=0; j< g.bs[i].rows; ++j){
            for (size_t k=0; k< g.bs[i].cols; ++k){
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }    
    }
    
}

void nn_learn(NN m, NN g, float learn_rate)
{
    
    for (size_t i=0; i< m.count; ++i){
    
        for (size_t row_x=0; row_x< m.ws[i].rows; ++row_x){
            for (size_t col_x=0; col_x < m.ws[i].cols; ++col_x){
                MAT_AT(m.ws[i], row_x, col_x) -= learn_rate * MAT_AT(g.ws[i], row_x, col_x);
            }
        }      

        for (size_t row_x=0; row_x< m.bs[i].rows; ++row_x){
            for (size_t col_x=0; col_x < m.bs[i].cols; ++col_x){
                MAT_AT(m.bs[i], row_x, col_x) -= learn_rate * MAT_AT(g.bs[i], row_x, col_x);
            }
        }
    }
}


void nn_print (NN nn, const char * layer)
{
    char buf[256];
    printf("%s : [\n", layer);
    Mat *ws = nn.ws;
    Mat *bs = nn.bs;
    for (size_t i=0; i < nn.count; ++i){
    snprintf(buf, sizeof buf, "ws%zu", i);
    mat_print(ws[i], buf, 4);
    snprintf(buf, sizeof buf, "bs%zu", i);
    mat_print(bs[i], buf, 4);
    }
    printf("]\n");
}


#endif


