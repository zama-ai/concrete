//
// Created by agnes leroy on 29/06/2021.
//
#include "concrete-ffi.h"
#include <cstdio>
#include <cassert>

void test_add(Variance v1, Variance v2) {
  Variance v3 = variance_add_u32(v1, v2);
  Variance v4 = variance_add_u64(v1, v2);
  assert(v3.val == pow(2., -24));
  assert(v4.val == pow(2., -24));

//Variance v5 = variance_add_several_u32(&variance_table);
//Variance v5 = variance_add_several_u64(&variance_table);
//assert(v5.val == 6);
}

void test_external_product(GlweDimension dimension,
                           PolynomialSize polynomial_size,
                           DecompositionBaseLog baseLog,
                           DecompositionLevelCount l_gadget,
                           Variance dispersion_ggsw,
                           Variance dispersion_qlwe) {

  Variance v1 = variance_external_product_u32_binary_key(
      dimension,
      polynomial_size,
      base_log,
      l_gadget,
      dispersion_ggsw,
      dispersion_glwe);
  assert(v1.val == 0);
  v1 = variance_external_product_u64_binary_key(
      dimension,
      polynomial_size,
      base_log,
      l_gadget,
      dispersion_ggsw,
      dispersion_glwe);
  assert(v1.val == 0);
  v1 = variance_external_product_u32_ternary_key(
      dimension,
      polynomial_size,
      base_log,
      l_gadget,
      dispersion_ggsw,
      dispersion_glwe);
  assert(v1.val == 0);
  v1 = variance_external_product_u64_ternary_key(
      dimension,
      polynomial_size,
      base_log,
      l_gadget,
      dispersion_ggsw,
      dispersion_glwe);
  assert(v1.val == 0);
}
int main (void) {
  Variance v1 = {pow(2., -25)};
  Variance v2 = {pow(2., -25)};
  test_add(v1, v2);
  GlweDimension dimension = 3;
  PolynomialSize polynomial_size = 1024;
  DecompositionBaseLog base_log = 7;
  DecompositionLevelCount l_gadget = 4;
  Variance v_ggsw = {pow(2., -38)};
  Variance v_glwe = {pow(2., -40)};
  test_external_product(dimension, polynomial_size, base_log, l_gadget,
                        v_ggsw, v_glwe);
}
