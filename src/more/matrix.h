#ifndef MATRIX_H
#define MATRIX_H

void clearCache (void);
float **allocArray(int, int);
void free_matrix(float **,int);
void printhelp(void);
void fillMatrix(float **,int,int);
void fillMatrixOne(float **,int ,int);
void fillMatrixZero(float **,int,int);
float **transposeMatrix(float **,int,int);
void printMatrix(float **,int,int);

#endif