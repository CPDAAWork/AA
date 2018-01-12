/*###############################################################################
#        Arquiteturas Avancadas
#################################################################################
#        Este software calcula a multiplicacão de matrizes quadradas
#        Bruno Chianca Ferreira
#        Daniel Rodrigues
#################################################################################*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "papi.h"
#include "more/matrix.h"
#include "more/papi_aux.h"

#define SIZE 32

long long unsigned timer1,timer2;

/*##############################################################################
 *
 *    NORMAL FORM
 *
 ###############################################################################*/

/***********************
 *    I-J-K NORMAL
 ***********************/
void dotMulti_ijk(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int i=0;i<_height;i++)
    for (int j=0;j<_width;j++)
      for (int k=0;k<_width;k++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}

/***********************
 *    I-K-J NORMAL
 ***********************/
void dotMulti_ikj(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int i=0;i<_height;i++)
    for (int k=0;k<_width;k++)
      for (int j=0;j<_width;j++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}

/***********************
 *    I-K-J NORMAL
 ***********************/
void dotMulti3(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int j=0;j<_height;j++)
    for (int k=0;k<_width;k++)
      for (int i=0;i<_width;i++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}

/*##############################################################################
 *
 *    TRANSPOSE FORM
 *
 ###############################################################################*/

/***********************
 *    I-J-K TRANSPOSE
 ***********************/
long long dotMulti_ijk_T(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  float ** tB;
  long long time_1=0;
  long long time_2=0;

  tB = transposeMatrix(_matrixB,_height,_width);

  time_1 = PAPI_get_real_usec();
  for (int i=0;i<_height;i++)
    for (int j=0;j<_width;j++)
      for (int k=0;k<_width;k++)
        _matrixC[i][j]+=_matrixA[i][k]*tB[j][k];
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_width);

  return time_2-time_1;
}

/***********************
 *    K-I-J TRANSPOSE
 ***********************/
long long dotMulti2_T(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  float **tA;
  long long time_1=0,time_2=0;

  tA = transposeMatrix(_matrixA,_height,_width);

  time_1 = PAPI_get_real_usec();
  for (int k=0;k<_height;k++)
    for (int i=0;i<_width;i++)
      for (int j=0;j<_width;j++)
        _matrixC[i][j]+=_matrixA[k][i]*_matrixB[k][j];
    time_2 = PAPI_get_real_usec();

  free_matrix(tA,_width);

  return (time_2-time_1);
}

/***********************
 *    J-K-I TRANSPOSE
 ***********************/
long long dotMulti3_T(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  float **tB, **tA;
  long long time_1=0,time_2=0;

  tA = transposeMatrix(_matrixA,_height,_width);
  tB = transposeMatrix(_matrixB,_height,_width);

  time_1 = PAPI_get_real_usec();
  for (int j=0;j<_height;j++)
    for (int k=0;k<_width;k++)
      for (int i=0;i<_width;i++)
        _matrixC[i][j]+=tA[k][i]*tB[j][k];
  time_2 = PAPI_get_real_usec();

  free_matrix(tA,_width);
  free_matrix(tB,_width);

  return (time_2-time_1);
}
/*##############################################################################
 *
 *    BLOCK FORM
 *
 ###############################################################################*/

/**************************************************
 *    I-J-K BLOCK
 **************************************************/
long long dotMulti_ijk_B_kj(float **_matrixA, float **_matrixB, float **_matrixC,int _block_size, int _height, int _width){
  float ** tB;
  long long time_1=0;
  long long time_2=0;

  tB = transposeMatrix(_matrixB,_height,_width);

  time_1 = PAPI_get_real_usec();
  int j_stop = 0;
  int k_stop = 0;
  double sum = 0;
  for(int kk = 0; kk<_height; kk+=_block_size){
    for(int jj = 0; jj<_height; jj+= _block_size){
      for(int i = 0; i< _height; i++){
        j_stop = jj+_block_size;
        for(int j = jj; j<j_stop; j++){
          k_stop = kk +_block_size;
          sum = _matrixC[i][j];
          for(int k = kk; k<k_stop; k++){
              sum+=_matrixA[i][k]*_matrixB[j][k];
          }
          _matrixC[i][j] = sum;
        }
      }
    }
  }
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_width);

  return time_2-time_1;
}

/**************************************************
 *    I-K-J BLOCK I-J-K
 **************************************************/
void multi_ijk(float **_matrixA, float **_matrixB, float **_matrixC,int i_start, int j_start, int k_start,int block_size){
  int i_stop = i_start+block_size;
  int j_stop = j_start+block_size;
  int k_stop = k_start+block_size;
  for (int i=i_start;i<i_stop;i++)
    for (int k=k_start;k<k_stop;k++)
      #pragma GCC ivdep
      for (int j=j_start;j<j_stop;j++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}
void dotMulti_block_ijk_ikj(float **_matrixA, float **_matrixB, float **_matrixC,int block_size,int size){
  for(int i=0; i<size; i+=block_size)
    for(int j=0; j<size; j+=block_size)
      for(int k=0; k<size; k+=block_size)
        multi_ijk(_matrixA,_matrixB,_matrixC,i,j,k,block_size);

}

/**************************************************
 *    K-I-J BLOCK I-K
 **************************************************/
void multi_kij(float ** _matrixA, float ** _matrixB, float ** _matrixC, int k_start, int i_start,int block, int size){
  int k_stop = k_start + block;
  int i_stop = i_start + block;
  for (int k=k_start;k<k_stop;++k){
    for (int i=i_start;i<i_stop;++i){
      float aik=_matrixA[i][k];
      #pragma GCC ivdep
      for (int j=0;j<size;j++){
        _matrixC[i][j]+=aik*_matrixB[k][j];
      }
    }
  }
}
void dotMulti2(float ** A, float ** B, float ** C,int block, int size){
  for(int i=0;i<size;i+=block){
    for(int k=0;k<size;k+=block){
      multi_kij(A,B,C,k,i,block,size);
    }
  } 
}

/*
// (k-i-j)
void mult2(float ** _matrixA, float ** _matrixB, float ** _matrixC, int k_start, int i_start,int block, int size){
  int k_stop = k_start + block;
  int i_stop = i_start + block;
  for (int k=k_start;k<k_stop;++k){
    for (int i=i_start;i<i_stop;i+=4){
      float aki=_matrixA[k][i];
      float aki2=_matrixA[k][i+1];
      float aki3=_matrixA[k][i+2];
      float aki4=_matrixA[k][i+4];
      #pragma GCC ivdep
      for (int j=0;j<size;++j){
        _matrixC[i][j]+=aki*_matrixB[k][j];
        _matrixC[i+1][j]+=aki2*_matrixB[k][j];
        _matrixC[i+2][j]+=aki3*_matrixB[k][j];
        _matrixC[i+3][j]+=aki4*_matrixB[k][j];
      }
    }
  }
}

void dotMulti2aki(float ** A, float ** B, float ** C,int block, int size){
  for(int i=0;i<size;i+=block){
    for(int k=0;k<size;k+=block){
        mult2(A,B,C,k,i,block,size);
    }
  } 
}
*/

int main(int argc, char *argv[]){

  int size = 32;
  int iter = 1;
  if(argc>1)
    size = atoi(argv[1]);
  if(argc>2)
    iter = atoi(argv[2]);

  float **A = allocArray(size,size);
  float **B = allocArray(size,size);
  float **C = allocArray(size,size);
  fillMatrix(A,size,size);
  fillMatrixOne(B,size,size);

  printf("\n\t ---> SIZE :: %d <---\n",size);

  printf("\nTYPE normal :: k-i-j vect+block\n");
  if(papi_init() == 0)
    return 0;

  for(int i = 0; i<iter; i++){
    clearCache();
    if(papi_start() == 0)
      return 0;

    int block = 8;

    timer1 = PAPI_get_real_usec();
    dotMulti2(A,B,C,block,size);
    timer2 = PAPI_get_real_usec();

    printf("%lld", (timer2-timer1));
    print_papi(size);
  }
  for (int h=0;h<size;h++){
    free(A[h]);
    free(B[h]);
    free(C[h]);
  }
}