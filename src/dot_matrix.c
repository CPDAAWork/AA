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

typedef unsigned long long type_time;
typedef float flop_type;

type_time (*func)(flop_type **, flop_type **,flop_type **,int,int);
FILE *fd;
int size = 32;
int iter = 1;

/*##############################################################################
 *
 *    NORMAL FORM
 *
 ###############################################################################*/

/***********************
 *    I-J-K NORMAL
 ***********************/
type_time dotMulti_ijk(flop_type **_matrixA, flop_type **_matrixB,
                       flop_type **_matrixC, int _height, int _width)
{
  type_time time_1=0;
  type_time time_2=0;
  int sum;
  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int i=0;i<_height;++i)
    for (int j=0;j<_width;++j){
      sum=0;
      for (int k=0;k<_width;++k)
        sum+=_matrixA[i][k]*_matrixB[k][j];
      _matrixC[i][j] = sum;
    }
  time_2 = PAPI_get_real_usec();

  return time_2-time_1;
}

/***********************
 *    I-K-J NORMAL
 ***********************/
type_time dotMulti_ikj(flop_type **_matrixA, flop_type **_matrixB,
                       flop_type **_matrixC, int _height, int _width)
{
  type_time time_1=0;
  type_time time_2=0;

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int i=0;i<_height;++i)
    for (int k=0;k<_width;++k)
      for (int j=0;j<_width;++j)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
    

  time_2 = PAPI_get_real_usec();

  return time_2-time_1;
}

/***********************
 *    J-K-I NORMAL
 ***********************/
type_time dotMulti_jki(flop_type **_matrixA, flop_type **_matrixB,
                       flop_type **_matrixC, int _height, int _width)
{
  type_time time_1=0;
  type_time time_2=0;

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int j=0;j<_height;++j)
    for (int k=0;k<_width;++k)
      for (int i=0;i<_width;++i)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
  time_2 = PAPI_get_real_usec();

  return time_2-time_1;
}

/*##############################################################################
 *
 *    TRANSPOSE FORM
 *
 ###############################################################################*/

/***********************
 *    I-J-K TRANSPOSE
 ***********************/
type_time dotMulti_ijk_T(flop_type **_matrixA, flop_type **_matrixB,
                         flop_type **_matrixC, int _height, int _width)
{
  flop_type ** tB;
  type_time time_1=0;
  type_time time_2=0;
  tB = transposeMatrix(_matrixB,_height,_width);
  int sum;

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int i=0;i<_height;++i)
    for (int j=0;j<_width;++j){
      sum=0;
      for (int k=0;k<_width;++k)
        sum+=_matrixA[i][k]*tB[j][k];
      _matrixC[i][j]=sum;
    }

  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_width);
  return time_2-time_1;
}

/***********************
 *    J-K-J TRANSPOSE
 ***********************/
type_time dotMulti_jki_T(flop_type **_matrixA, flop_type **_matrixB,
                         flop_type **_matrixC, int _height, int _width)
{
  flop_type **tB, **tA, **tC;
  type_time time_1=0,time_2=0;
  tA = transposeMatrix(_matrixA,_height,_width);
  tB = transposeMatrix(_matrixB,_height,_width);
  tC = transposeMatrix(_matrixC,_height,_width);

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int j=0;j<_height;++j)
    for (int k=0;k<_width;++k)
      for (int i=0;i<_width;++i)
        _matrixC[i][j]+=tA[k][i]*tB[j][k];
  time_2 = PAPI_get_real_usec();

  flop_type ** aux = _matrixC;_matrixC = transposeMatrix(tC,_height,_width);
  tC = aux;
  free_matrix(tA,_width);free_matrix(tB,_width);
  return (time_2-time_1);
}

/***********************
 *    K-I-J TRANSPOSE
 ***********************/
/*
type_time dotMulti_kij_T(flop_type **_matrixA, flop_type **_matrixB, flop_type **_matrixC, int _height, int _width){
  flop_type **tA;
  type_time time_1=0,time_2=0;

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
*/

/*##############################################################################
 *
 *    BLOCK FORM
 *
 ###############################################################################*/
type_time bijk(flop_type **A, flop_type **B, flop_type **C, int _block_size, int _size){
  int i, j, k, kk, jj; 
  float sum;
  flop_type ** tB;
  type_time time_1=0;
  type_time time_2=0;
  tB = transposeMatrix(B,_size,_size);

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (kk = 0; kk < _size; kk += _block_size) {
      for (jj = 0; jj < _size; jj += _block_size) {
        for (i = 0; i < _size; ++i) {
          for (j = jj; j < jj + _block_size; ++j) {
            sum = C[i][j];
            for (k = kk; k < kk + _block_size; ++k) {
              sum += A[i][k]*tB[j][k];
          }
          C[i][j] = sum;
        }
      }
    }
  }
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_size);
  return time_2-time_1;
}

void multi_ijk_block(flop_type **_matrixA, flop_type **_matrixB, flop_type **_matrixC,
                int i_start, int j_start, int k_start,int _block_size,
                int _size)
{
  int i_stop = i_start+_block_size;
  int j_stop = j_start+_block_size;
  for (int i=i_start;i<i_stop;++i)
    for (int j=j_start;j<j_stop;++j)
      for (int k=k_start;k<_size;++k)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[j][k];
}

type_time dotMulti_ijk_B_ijk(flop_type **_matrixA, flop_type **_matrixB,
                             flop_type **_matrixC,int _block_size, int _size)
{
  flop_type ** tB;
  type_time time_1=0;
  type_time time_2=0;
  tB = transposeMatrix(_matrixB,_size,_size);

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for(int j = 0; j<_size; j+= _block_size)
    for(int i = 0; i<_size; i+= _block_size)
      for(int k = 0; k<_size; k+= _block_size)
        multi_ijk_block(_matrixA,_matrixB,_matrixC,i,j,0,_block_size,_size);
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_size);
  return time_2-time_1;
}

/*##############################################################################
 *
 *    BLOCK FORM AND VECT
 *
 ###############################################################################*/

//NOT VECT
void multi_ijk(flop_type **_matrixA, flop_type **_matrixB, flop_type **_matrixC,
                int i_start, int j_start, int k_start,int _block_size,
                int _size)
{
  int i_stop = i_start+_block_size;
  int j_stop = j_start+_block_size;
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int i=i_start;i<i_stop;++i)
    for (int j=j_start;j<j_stop;++j)
      #ifdef VEC
        #pragma GCC ivdep
      #endif
      for (int k=0;k<_size;++k)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[j][k];
}
//VECT
void multi_kij(flop_type ** _matrixA, flop_type ** _matrixB, flop_type ** _matrixC,
                int i_start, int j_start,int k_start,int _block_size,
                int _size)
{
  int k_stop = k_start + _block_size;
  int i_stop = i_start + _block_size;
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int k=k_start;k<k_stop;++k){
    for (int i=i_start;i<i_stop;++i){
      flop_type aik=_matrixA[i][k];
      #ifdef VEC
        #pragma GCC ivdep
      #endif
      for (int j=0;j<_size;++j){
        _matrixC[i][j]+=aik*_matrixB[k][j];
      }
    }
  }
}
//VECT
void multi_ikj(flop_type ** _matrixA, flop_type ** _matrixB, flop_type ** _matrixC,
                int i_start, int j_start,int k_start,int _block_size,
                int _size)
{
  int k_stop = k_start + _block_size;
  int i_stop = i_start + _block_size;
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int i=i_start;i<i_stop;++i)
    for (int k=k_start;k<k_stop;++k){
      flop_type aik=_matrixA[i][k];
      #ifdef VEC
        #pragma GCC ivdep
      #endif
      for (int j=0;j<_size;++j)
        _matrixC[i][j]+=aik*_matrixB[k][j];
    }
}

/**************************************************
 *    I-J-K BLOCK
 **************************************************/
type_time dotMulti_ji_B_ijk(flop_type **_matrixA, flop_type **_matrixB,
                             flop_type **_matrixC,int _block_size, int _size)
{
  flop_type ** tB;
  type_time time_1=0;
  type_time time_2=0;
  tB = transposeMatrix(_matrixB,_size,_size);

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for(int j = 0; j<_size; j+= _block_size)
    for(int i = 0; i<_size; i+= _block_size)
      multi_ijk(_matrixA,_matrixB,_matrixC,i,j,0,_block_size,_size);
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_size);
  return time_2-time_1;
}

/**************************************************
 *    I-K-J BLOCK I-J-K
 **************************************************/
type_time dotMulti_ij_B_ikj(flop_type **_matrixA, flop_type **_matrixB,
                             flop_type **_matrixC,int _block_size,int _size)
{
  flop_type ** tB;
  type_time time_1=0;
  type_time time_2=0;
  tB = transposeMatrix(_matrixB,_size,_size);

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for(int i=0; i<_size; i+=_block_size)
    for(int j=0; j<_size; j+=_block_size)
      multi_ijk(_matrixA,_matrixB,_matrixC,i,j,0,_block_size,_size);
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_size);
  return time_2-time_1;
}

type_time dotMulti_ik_B_ikj(flop_type **_matrixA, flop_type **_matrixB,
                             flop_type **_matrixC,int _block_size, int _size)
{
  flop_type ** tB;
  type_time time_1=0;
  type_time time_2=0;
  tB = transposeMatrix(_matrixB,_size,_size);

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for(int i = 0; i<_size; i+= _block_size)
    for(int k = 0; k<_size; k+= _block_size)
      multi_ikj(_matrixA,_matrixB,_matrixC,i,0,k,_block_size,_size);
  time_2 = PAPI_get_real_usec();

  free_matrix(tB,_size);
  return time_2-time_1;
}

/**************************************************
 *    K-I-J BLOCK I-K
 **************************************************/
type_time dotMulti_ik_B_kij(flop_type ** A, flop_type ** B, flop_type ** C,
                            int _block_size, int _size)
{
  type_time time_1=0;
  type_time time_2=0;
 
  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for(int i=0; i<_size; i+=_block_size)
    for(int k=0; k<_size; k+=_block_size)
      multi_kij(A,B,C,i,0,k,_block_size,_size);
  time_2 = PAPI_get_real_usec();

  return time_2-time_1;
}

/*
// (k-i-j)
void mult2(flop_type ** _matrixA, flop_type ** _matrixB, flop_type ** _matrixC, int k_start, int i_start,int block, int size){
  int k_stop = k_start + block;
  int i_stop = i_start + block;
  for (int k=k_start;k<k_stop;++k){
    for (int i=i_start;i<i_stop;i+=4){
      flop_type aki=_matrixA[k][i];
      flop_type aki2=_matrixA[k][i+1];
      flop_type aki3=_matrixA[k][i+2];
      flop_type aki4=_matrixA[k][i+4];
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

void dotMulti2aki(flop_type ** A, flop_type ** B, flop_type ** C,int block, int size){
  for(int i=0;i<size;i+=block){
    for(int k=0;k<size;k+=block){
        mult2(A,B,C,k,i,block,size);
    }
  } 
}
*/
/*##############################################################################
 *
 *    SIMPLE VECT
 *
 ###############################################################################*/

type_time dotMulti_ikj_V(flop_type **_matrixA, flop_type **_matrixB,
                         flop_type **_matrixC, int _height, int _width)
{
  type_time time_1=0;
  type_time time_2=0;

  time_1 = PAPI_get_real_usec();
  #ifdef OMP
    #pragma omp parallel for
  #endif
  for (int i=0;i<_height;++i)
    for (int k=0;k<_width;++k)
      #ifdef VEC
        #pragma GCC ivdep
      #endif
      for (int j=0;j<_width;++j)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
  time_2 = PAPI_get_real_usec();

  return time_2-time_1;
}

int exec_Function(flop_type **_matrixA, flop_type **_matrixB,
                   flop_type **_matrixC, int _height, int _width)
{
  fillMatrixZero(_matrixC,_width,_width);
  type_time timer;
  if(papi_init() == 0)
    return -1;

  for(int i = 0; i<iter; i++){
    clearCache();
    if(papi_start() == 0)
      return -1;

    timer=func(_matrixA,_matrixB,_matrixC,_height,_width);

    printf("%lld", (timer));
    print_papi(_width,fd);
  }
  return 1;
}

int main(int argc, char *argv[]){
  int diff = 0;
  char repFile[] = "relatorio.csv";

  if(argc>1)
    size = atoi(argv[1]);
  if(argc>2)
    iter = atoi(argv[2]);
  if (fd==NULL){
    fprintf(stderr, "Problem trying to open file for saving report!\n");
    exit(1);
  }
  flop_type **A = allocArray(size,size);
  flop_type **B = allocArray(size,size);
  flop_type **C = allocArray(size,size);
  flop_type **C2 = allocArray(size,size);

  fillMatrix(A,size,size);
  fillMatrixOne(B,size,size);

  printf("SIZE %d\n",size);
  printf("Simple Vec\n");
  func = &dotMulti_ikj_V;
  exec_Function(A,B,C2,size,size);
  //printMatrix(C2,size,size);

  printf("Block Not Vec ik-kij\n");
  func = &dotMulti_ik_B_kij;
  exec_Function(A,B,C,32,size);
  //printMatrix(C,size,size);

  //diff = comp_matrices(C,C2,size);
  printf("Diff %d\n",diff);

  printf("Block Vec ik-ikj\n");
  func = &bijk;
  exec_Function(A,B,C,32,size);
  //printMatrix(C,size,size);

  //diff = comp_matrices(C,C2,size);
  printf("Diff %d\n",diff);

  for (int h=0;h<size;h++){
    free(A[h]);
    free(B[h]);
    free(C[h]);
  }
}