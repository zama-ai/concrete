#ifndef CONCRETE_NUM_COMPLEX_H
#define CONCRETE_NUM_COMPLEX_H

#ifdef __cplusplus
#include <complex>

typedef std::complex<double> c64;
#endif

// Needed for tests written in Zig because Zig does not have interability with // C99 complex types.
#ifndef __cplusplus
struct c_double_complex {
  double c[2];
};

typedef struct c_double_complex c64;
#endif

#endif
