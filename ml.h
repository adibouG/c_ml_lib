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


typedef struct {
	Mat mat[];
	Activation a;	
} NNLayer;

float rand_float(void);
//activation fn
float sigmf(float x);
float reluf(float x);
//float sinf(float x);
//matrice fn
Mat mat_alloc(size_t rows, size_t cols);
Mat mat_getrow(Mat m, size_t row);
Mat mat_getsubmat(Mat m, size_t from_row, size_t to_row, size_t from_col, size_t to_col);

Mat mat_sub(Mat m, size_t stride, size_t start);

void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_rand(Mat m, float min, float max);
void mat_print(Mat m, const char * layer);
void mat_fill(Mat m, float val);

void mat_sin(Mat m);
void mat_sig(Mat m);
void mat_relu(Mat m);

void mat_activate_fn(Mat m, Activation f);

#define MAT_PRINT(m) mat_print(m, #m)
#define MAT_AT(m, r, c) (m).es[(r)*(m).stride + (c)]

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

Mat mat_sub(Mat m, size_t stride, size_t start)
{
    /*
     * float d[] = {
     * 	0,0,0,
     *  0,1,1,
     *  1,0,1,
     *  1,1,1
     * };
     * 	Mat sub_i = {rows=4, cols=2,stride=3, es=&d[0]}
     * 	Mat sub_o = {rows=4, cols=1,stride=3, es=&d[2]}
     */
     Mat dest;
    mat_copy(dest, m);
    for (size_t r=0; r < m.rows; ++r){
        for (size_t c=0; c < m.cols; ++c){
            MAT_AT(dest, r, c) = MAT_AT(m, f + r, start + c);
        }
    }
    return e;
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

void mat_print(Mat m, const char * layer)
{
	printf("%s:\n", layer);
	for (size_t r=0; r < m.rows; ++r){
		printf("\t");
		for (size_t c=0; c < m.cols; ++c){
			printf("%f  ",MAT_AT(m, r, c));
		}
		printf("\n");
	}
	printf("]\n");
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



#endif


