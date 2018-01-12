/*###############################################################################
#        Arquiteturas Avancadas
#################################################################################
#        Este software calcula a multiplicacão de matrizes quadradas
#        Bruno Chianca Ferreira
#        Daniel Rodrigues
#################################################################################*/

#include <stdio.h>
#include <stdlib.h>

#define SIZE_MATRIX 30000000

double clearcache [SIZE_MATRIX];

void clearCache (void) {
    for (unsigned i = 0; i < SIZE_MATRIX; ++i)
        clearcache[i] = i;
}

float **allocArray(int _height, int _width){
  float ** __restrict__ _matrix __attribute__ ((aligned(64))) = (float**) malloc(sizeof(float*)*_height);
  for (int h=0;h<_height;h++){
    _matrix[h] = (float*) malloc(sizeof(float)*_width);
  }
  return _matrix;
}

void free_matrix(float ** _matrix,int _lines){
  if(_matrix != NULL){
      for(int i=0; i<_lines; i++){
        if(_matrix[i]!= NULL)
          free(_matrix[i]);
        else
          break;
      }
      free(_matrix);
    }
}

void printhelp(){
    printf("Calculates matrices multiplication\n");
    printf("Usage: dotmatrix [SIZE]\n");
  printf("If size is not specified, taken as default 5\n");
    exit(1);
}

void fillMatrix(float **_matrix, int _height, int _width){
  for (int h=0;h<_height;h++){
      for (int w=0;w<_width;w++){
        _matrix[h][w]=((float) rand()) / ((float) RAND_MAX);
      }
  }
}

void fillMatrixOne(float **_matrix, int _height, int _width){
  for (int h=0;h<_height;h++){
      for (int w=0;w<_width;w++){
        _matrix[h][w]=1;
      }
  }
}

void fillMatrixZero(float **_matrix, int _height, int _width){
  for (int h=0;h<_height;h++){
      for (int w=0;w<_width;w++){
        _matrix[h][w]=0;
      }
  }
}

float ** transposeMatrix(float **_matrix, int _height, int _width){
  float ** transpose = allocArray(_width,_height);

  for(int i = 0; i<_height; i++)
    for(int j = 0; j<_width; j++)
      transpose[j][i]=_matrix[i][j];

  return transpose;
}

void printMatrix(float **_matrix, int _height, int _width){
  for (int h=0;h<_height;h++){
      for (int w=0;w<_width;w++){
        printf("%f ", _matrix[h][w]);
      }
      printf("\n");
  }
}