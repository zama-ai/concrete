//
// Created by agnes leroy on 29/06/2021.
//
#include "concrete-ffi.h"
#include <cstdio>

int main (void) {
  Variance v1 = {1.};
  Variance v2 = {1.};
  Variance v3 = npe_add_ciphertexts_variance_variance(v1, v2);
  printf("Result variance = %f\n", v3.variance);
}
