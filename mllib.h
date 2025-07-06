#ifndef MLLIB_H
#define MLLIB_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//#define STB_IMAGE_IMPLEMENTATION
//#include "./stb_image.h"

#ifndef ML_MALLOC
#define ML_MALLOC malloc
#endif

#ifndef ML_ASSERT
#include <assert.h>
#define ML_ASSERT assert
#endif

typedef enum {
    Activ_None = 0,
    Activ_Sig,
    Activ_Rel,
    Activ_Sin,
    Activ_count
} ActivationFn;


typedef enum {
    ALGO_None = 0,
    ALGO_FiniteDif, // FiniteDifferentiation
    ALGO_BackProp, // Backprop
    ALGO_SGD, // Stochastic Gradient Descent
    ALGO_count
} ALGORITHM;

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float_t *es;
} Mat;

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float_t *es;
} Mat_View;

typedef struct {
    size_t count;
    size_t capacity;
    size_t * items;
} Arch;


float rand_float(void);
//activation fn
float sigmf(float x);
float reluf(float x);
//float sinf(float x);
//matrice fn
Mat mat_alloc(size_t rows, size_t cols);
Mat mat_getrow(Mat m, size_t row);
Mat mat_getsubmat(Mat m, size_t from_row, size_t to_row, size_t from_col, size_t to_col);

void mat_shuffle_rows(Mat m);
void mat_shuffle_rows_sync(Mat m, Mat m2);
void mat_split_data(Mat m, int size, Mat * batch_list);
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

//  void mat_activate_fn(Mat m, Activation f);

int mat_save(const char * filename, Mat m);
Mat mat_load(const char * filepath) ;
void print_img(const char * filename);
Mat img_to_mat(const char * filename);


#define MAT_PRINT(m) mat_print(m, #m, 0);
#define MAT_AT(m, r, c) (m).es[(r)* (m).stride + (c)]
#define MAT_Wstride_AT(m, r, c) (m).es[(r)*(m).stride + (c)]



typedef struct {
    Mat * mat;
    ActivationFn a;
} NNLayer;


typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as; // amount activations count +1
    ALGORITHM method;  // "sgd", "adam", etc.
    ActivationFn activation;  // "sigm", "relu", etc.
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

void mat_activate_fn(Mat a, ActivationFn f)
{
    for (size_t r=0; r< a.rows; ++r){
        for (size_t c=0; c< a.cols; ++c){
            switch (f){
                case Activ_Sig:
                    MAT_AT(a, r, c) = sigmf(MAT_AT(a, r, c));
                    break;
                case Activ_Rel:
                    MAT_AT(a, r, c) = reluf(MAT_AT(a, r, c));
                    break;
                case Activ_Sin:
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

    size_t nrows = (torow - fromrow) + 1;
    size_t ncols = (tocol - fromcol) + 1;

    Mat e = mat_alloc(nrows, ncols);

    for (size_t r=0; r < nrows; ++r){
        for (size_t c=0; c < ncols; ++c){
            MAT_AT(e, r, c) = MAT_AT(m, fromrow + r, fromcol + c);
        }
    }
    return e;
}

Mat mat_subview(Mat m, size_t start, size_t end)
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
        .cols   = (end - start) + 1 ,
        .stride = m.stride,
        .es     = &MAT_AT(m, 0, start)
    };
}



void mat_shuffle_rows(Mat m)
{
    for (size_t r=0; r < m.rows; ++r){
        size_t r2 = r + rand() % (m.rows - r)  ; //r2 = r + rand() % m.rows; //  rand() % m.rows
        if (r == r2) {
            continue; // no need to swap with itself
        }
        // swap rows r and r2, column by column
        for (size_t c=0; c < m.cols; ++c){
            float tmp = MAT_AT(m, r, c);
            MAT_AT(m, r, c) = MAT_AT(m, r2, c);
            MAT_AT(m, r2, c) = tmp;
        }
    }
}


void mat_shuffle_rows_sync(Mat m, Mat m2)
{
    ML_ASSERT(m.rows == m2.rows) ;
    for (size_t r=0; r < m.rows; ++r){
        size_t r2 = r + rand() % (m.rows - r)  ; //r2 = r + rand() % m.rows; //  rand() % m.rows
        if (r == r2) {
            continue; // no need to swap with itself
        }
        float tmp;
        // swap rows r and r2, column by column
        for (size_t c=0; c < m.cols; ++c){
            tmp = MAT_AT(m, r, c);
            // swap rows r and r2, column by column
            MAT_AT(m, r, c) = MAT_AT(m, r2, c);
            MAT_AT(m, r, c) = MAT_AT(m, r2, c);
            MAT_AT(m, r2, c) = tmp;
        }
        for (size_t c=0; c < m2.cols; ++c){
            tmp = MAT_AT(m2, r, c);
            MAT_AT(m2, r, c) = MAT_AT(m2, r2, c);
            MAT_AT(m2, r, c) = MAT_AT(m2, r2, c);
            MAT_AT(m2, r2, c) = tmp;
        }
    }
}
/*  6/4 = 1  r2
    [1 2 3 
    4 5 6

    7 8 9
    10 11 12
     
    13 14 15
    16 17 18]

*/
void mat_split_data(Mat m, int size, Mat * batch_list)
{
    ML_ASSERT(m.rows > 0 && m.cols > 0);
    ML_ASSERT(size > 0 && size <= m.rows);

    if (size == 1 || size == m.rows) {
        batch_list[0] = m; // no split, no sgd, just return the original matrix
        return;
    }
   
    if (m.rows % size == 0) { // rows must be divisible by size
        int count = m.rows / size;
        for (int i=0; i < count; ++i){
            size_t start_row = i % m.rows; // size; // i * (m.rows / size);
            size_t end_row = start_row + size - 1;
            batch_list[i] = mat_subview(m, start_row, end_row);
        }
    }
    else {
        fprintf(stderr, "mat_split_data: rows %zu not divisible by size %d\n", m.rows, size);
        ML_ASSERT(0); // rows must be divisible by size
    }   
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




int mat_save(const char * filename, Mat m)
{
    const char * magic = "mat.data";
    FILE * data_out = fopen(filename,"wb");
    if (data_out == NULL) {
        fprintf(stderr, "Could not open file %s\n", filename);
        return 1;
    }
    fwrite(magic, strlen(magic),    1, data_out);
    fwrite(&m.rows, sizeof(m.rows),  1, data_out);
    fwrite(&m.cols, sizeof(m.cols),  1, data_out);

    for (size_t i = 0; i <m.rows; ++i){
        size_t n = fwrite(&MAT_AT(m, i, 0), sizeof(*m.es), m.cols, data_out);
        while (n < m.cols && !ferror(data_out)){
            size_t k = fwrite(m.es + n, sizeof(*m.es), m.cols - n, data_out);
            n += k;
        }
    }
    if (ferror(data_out)) {
        fprintf(stderr, "write error on file %s\n", filename);
        fclose(data_out);
        return 1;
    }

    fclose(data_out);
    fflush(stdout);

    printf("mat_save()........Results view:.....\n");
    printf("rows: %zu, cols: %zu\n", m.rows, m.cols);
    printf("size: %zu bytes\n", sizeof(*m.es)*m.rows*m.cols + sizeof(magic) + sizeof(m.rows) + sizeof(m.cols));
    printf("magic: %s\n", magic);
    printf("data saved to %s\n", filename);
    printf("mat_save()..................:.....\n");

    return 0;
}

Mat mat_load(const char * filepath)
{
    u_int64_t magic;
    FILE * data_in = fopen(filepath,"rb");
    Mat m;
    if (data_in == NULL) {
        fprintf(stderr, "Could not open file %s\n",filepath);
        m = mat_alloc(0, 0);
        fclose(data_in);
        return m;
    }

    fread(&magic, sizeof(magic),1, data_in);
  
    uint64_t magic_le = ((magic << 56) | 
            (((magic >> 48) << 56) >> 48) |
            (((magic >> 40) << 56) >> 40) |
            (((magic >> 32) << 56) >> 32) |
            (((magic >> 24) << 56) >> 24) |
            (((magic >> 16) << 56) >> 16) |
            (((magic << 16) >> 56) << 16) |
            (((magic << 24) >> 56) << 24) |
            (((magic << 32) >> 56) << 32) |
            (((magic << 40) >> 56) << 40) |
            (((magic << 48) >> 56) << 48) |
        (magic >> 56));
    printf("magic: 0x%llx, magic_le: 0x%llx\n", magic, magic_le);    
    if (magic != 0x6d61742e64617461 &&
         magic_le != 0x6d61742e64617461) 
    {          
        fprintf (stderr, "magic mismatch 0x%llx != 0x6d61742e64617461\n", magic);
        fprintf (stderr, "magic_le (reversed, little endianess) mismatch too 0x%llx != 0x6d61742e64617461\n", magic_le);
        fprintf(stderr, "file %s is not a valid .mat file\n", filepath);        
        fclose(data_in);
        m = mat_alloc(0, 0);
        return m;
    }
    
    size_t rows, cols;
    fread(&rows, sizeof (rows), 1 , data_in);
    fread(&cols, sizeof (cols), 1 , data_in);
    m = mat_alloc(rows, cols);
    size_t n = fread(m.es, sizeof (*m.es), rows * cols, data_in);
    while( n < cols*rows )
    {
        size_t k = fread(m.es, sizeof (*m.es ) + n, rows * cols -n, data_in);
        n += k;
    }
    if (ferror(data_in)) {
        fprintf(stderr, "read error on file %s\n", filepath);
        m = mat_alloc(0, 0);
        return m;
    }
    fclose(data_in);
    fflush(stdout);
    printf("Matrix data loaded from %s\n", filepath);
    return m;
}

/*
Mat img_to_mat(const char * filename)
{
    int img_w, img_h, img_c;
    uint8_t * pixel_data = (uint8_t *) stbi_load(filename, &img_w, &img_h, &img_c, 0);
    printf("%s size %d x %d , %d bits\n", filename, img_w, img_h, img_c*8 );

    Mat m = mat_alloc(img_w*img_h, 3) ;

    for (int x=0; x < img_w; ++x){
        for (int y=0; y < img_h; ++y){
            size_t i = (y * img_w + x ) ;
            MAT_AT(m, i , 0) = (float) x/(img_w -1) ;
            MAT_AT(m, i , 1) = (float) y/(img_h -1) ;
            MAT_AT(m, i , 2) = pixel_data[i]/255.f ;
        }
    }
    return m;
}

void print_img(const char * filename)
{
    int img_w, img_h, img_c;
    uint8_t * pixel_data = (uint8_t *) stbi_load(filename, &img_w, &img_h, &img_c, 0);
    printf("%s size %d x %d , %d bits\n", filename, img_w, img_h, img_c*8 );

    for (int x=0; x < img_w; ++x){
        for (int y=0; y < img_h; ++y){
            size_t i = (y * img_w + x ) ;
            printf("%3u ", pixel_data[i]);
        }
        printf("\n");
    }
}
*/
///////////////////////////////////////////////////////


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

void nn_zero(NN n)
{
    for (size_t i=0; i < n.count; ++i){
            mat_fill(n.ws[i], 0);
            mat_fill(n.bs[i], 0);
            mat_fill(n.as[i], 0);
    }
    mat_fill(n.as[n.count], 0);
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


