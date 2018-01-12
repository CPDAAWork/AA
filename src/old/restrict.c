
float **allocArray(int _height, int _width){
  float ** __restrict__ _matrix __attribute__ ((aligned(64))) = (float**) malloc(sizeof(float*)*_height);
  for (int h=0;h<_height;h++){
    _matrix[h] = (float*) malloc(sizeof(float)*_width);
  }
  return _matrix;
}