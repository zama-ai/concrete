#ifndef C64_OPS_H
#define C64_OPS_H

#include "concrete-cpu.h"
#include <math.h>

static inline c64 c64_mul(c64 a, c64 b) {
  c64 ret;

  ret.c[0] = a.c[0] * b.c[0] - a.c[1] * b.c[1];
  ret.c[1] = a.c[0] * b.c[1] + a.c[1] * b.c[0];

  return ret;
}

static inline c64 c64_add(c64 a, c64 b) {
  c64 ret;

  ret.c[0] = a.c[0] + b.c[0];
  ret.c[1] = a.c[1] + b.c[1];

  return ret;
}

static inline c64 c64_sub(c64 a, c64 b) {
  c64 ret;

  ret.c[0] = a.c[0] - b.c[0];
  ret.c[1] = a.c[1] - b.c[1];

  return ret;
}

static inline c64 c64_build(double r, double i) {
  c64 ret;
  ret.c[0] = r;
  ret.c[1] = i;

  return ret;
}

static inline c64 c64_conj(c64 a) { return c64_build(a.c[0], -a.c[1]); }
static inline c64 c64_inv(c64 a) { return c64_build(-a.c[0], -a.c[1]); }
static inline c64 c64_rdiv(c64 a, double f) {
  return c64_build(a.c[0] / f, a.c[1] / f);
}

static inline c64 c64_i(void) { return c64_build(0, 1); }

static inline c64 c64_exp(c64 v) {
  c64 ret;

  ret.c[0] = exp(v.c[0]) * cos(v.c[1]);
  ret.c[1] = exp(v.c[0]) * sin(v.c[1]);

  return ret;
}

#endif
