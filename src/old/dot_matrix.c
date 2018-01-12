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

#define NUM_EVENTS 3
#define SIZE 32

long long values[NUM_EVENTS], min_values[NUM_EVENTS];
long long unsigned timer1,timer2;
int Events[NUM_EVENTS] = { PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_TCM};
double clearcache [30000000];
int num_hwcntrs;

void print_papi();
int papi_start ();
int papi_init();

void clearCache (void) {
    for (unsigned i = 0; i < 30000000; ++i)
        clearcache[i] = i;
}

float **allocArray(int _height, int _width){
  float **_matrix = (float**) malloc(sizeof(float*)*_height);
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

/* 2.2 BEGIN */
//(i-j-k)
void dotMulti(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int i=0;i<_height;i++)
    for (int j=0;j<_width;j++)
      for (int k=0;k<_width;k++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}

//(i-j-k) Transpose
long long dotMulti_T(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
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
/* 2.2 END */

/* 2.3 BEGIN */
/* row by row */
// (1)(i-k-j)
void dotMulti1(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int i=0;i<_height;i++)
    for (int k=0;k<_width;k++)
      for (int j=0;j<_width;j++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}

// (k-i-j)
void dotMulti2(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int k=0;k<_height;k++)
    for (int i=0;i<_width;i++)
      for (int j=0;j<_width;j++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];

}

// (k-i-j) Transpose
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

/* Column by Column*/
// (2)(j-k-i)
void dotMulti3(float **_matrixA, float **_matrixB, float **_matrixC, int _height, int _width){
  for (int j=0;j<_height;j++)
    for (int k=0;k<_width;k++)
      for (int i=0;i<_width;i++)
        _matrixC[i][j]+=_matrixA[i][k]*_matrixB[k][j];
}

// (2)(j-k-i) Transpose
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
/* 2.3 END */

int main(int argc, char *argv[]){

  int size = 32;
  int iter = 20;
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

  printf("\nTYPE :: i-j-k\n");
  if(papi_init() == 0)
    return 0;

  for(int i = 0; i<iter; i++){
    clearCache();
    if(papi_start() == 0)
      return 0;

    timer1 = PAPI_get_real_usec();
    dotMulti(A,B,C,size,size);
    timer2 = PAPI_get_real_usec();

    //printf("%lld", (timer2-timer1));
    print_papi();
  }

  printf("\nTYPE :: i-k-j\n");
  if(papi_init() == 0)
    return 0;

  for(int i = 0; i<iter; i++){
    clearCache();
    if(papi_start() == 0)
      return 0;

    timer1 = PAPI_get_real_usec();
    dotMulti1(A,B,C,size,size);
    timer2 = PAPI_get_real_usec();

    //printf("%lld", (timer2-timer1));
    print_papi();
  }

  printf("\nTYPE :: j-k-i\n");
  if(papi_init() == 0)
    return 0;

  for(int i = 0; i<iter; i++){
    clearCache();
    if(papi_start() == 0)
      return 0;

    timer1 = PAPI_get_real_usec();
    dotMulti3(A,B,C,size,size);
    timer2 = PAPI_get_real_usec();

    //printf("%lld", (timer2-timer1));
    print_papi();
  }

  printf("\nTYPE :: k-i-j\n");
  if(papi_init() == 0)
    return 0;

  for(int i = 0; i<iter; i++){
    clearCache();
    if(papi_start() == 0)
      return 0;

    timer1 = PAPI_get_real_usec();
    dotMulti2(A,B,C,size,size);
    timer2 = PAPI_get_real_usec();

    //printf("%lld", (timer2-timer1));
    print_papi();
  }

  for (int h=0;h<size;h++){
    free(A[h]);
    free(B[h]);
    free(C[h]);
  }
}


int papi_init (){

  PAPI_library_init (PAPI_VER_CURRENT);
  /* Get the number of hardware counters available */
  if ((num_hwcntrs = PAPI_num_counters()) <= PAPI_OK)  {
    printf ("PAPI error getting number of available hardware counters!\n");
    return 0;
  }    
  // We will be using at most NUM_EVENTS counters
  if (num_hwcntrs >= NUM_EVENTS) {
    num_hwcntrs = NUM_EVENTS;
  } else {
    printf ("Error: there aren't enough counters to monitor %d events!\n", NUM_EVENTS);
    return 0;
  }
  return 1;
}

int papi_start() {
  /* Start counting events */
  if (PAPI_start_counters(Events, num_hwcntrs) != PAPI_OK) {
    printf ("PAPI error starting counters %d %d!\n",num_hwcntrs, NUM_EVENTS);
    return 0;
  }
  return 1;
}


void print_papi(){
    /* Stop counting events */
    if (PAPI_stop_counters(values, NUM_EVENTS) != PAPI_OK) {
      printf ("PAPI error stoping counters!\n");
    }
    //Size matrix C
    int total_elements = SIZE*SIZE;

    for (int i=0 ; i< NUM_EVENTS ; i++) 
      min_values[i] = values[i];


    // output PAPI counters' values
    for (int i=0 ; i< NUM_EVENTS ; i++) {
      char EventCodeStr[PAPI_MAX_STR_LEN];

      if (PAPI_event_code_to_name(Events[i], EventCodeStr) == PAPI_OK) {
        printf (";%lld", min_values[i]);
      } else {
        printf ("PAPI UNKNOWN EVENT = %lld\n", min_values[i]);
      }
    }
    printf("\n");

#if NUM_EVENTS >1
    // evaluate CPI and Texec here
      if ((Events[0] == PAPI_TOT_CYC) && (Events[1] == PAPI_TOT_INS)) {
        float CPI = ((float) min_values[0]) / ((float) min_values[1]);
        float CPE = ((float) min_values[0]) / ((float) total_elements);
        long long Texec = (long long) (((float)min_values[0])/2e3);
        //printf ("%lld\n",Texec);
      }
#endif
}