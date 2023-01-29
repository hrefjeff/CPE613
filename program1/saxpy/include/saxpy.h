#ifndef __saxpy_h__
#define __saxpy_h__

#ifdef __cplusplus
extern "C" {
#endif

void saxpy (
  int       n,
  float     alpha,
  float *   dev_x,
  int       incx,
  float *   dev_y,
  int       incy
);

#ifdef __cplusplus
}
#endif

#endif
