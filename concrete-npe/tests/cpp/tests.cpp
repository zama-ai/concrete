//
// Created by agnes leroy on 29/06/2021.
//
#include "concrete-npe.h"
#include <cstdio>

int main (void) {
  double v1 = 1.;
  double v2 = 1.;
  double v3 = npe_ffi::add_ciphertexts_variance_variance(v1, v2);
  printf("Result variance = %f\n", v3);
}
