#ifndef GPU_BOOTSTRAP_COMMON_CUH
#define GPU_BOOTSTRAP_COMMON_CUH

#include <cassert>
#include <cstdint>
#include <cstdio>

#define SNT 1
#define dPI 6.283185307179586231995926937088

using sTorus = int32_t;
// using Torus = uint32_t;
using sTorus = int32_t;
using u32 = uint32_t;
using i32 = int32_t;

//--------------------------------------------------
// Basic double2 operations

__device__ inline double2 conjugate(const double2 num) {
  double2 res;
  res.x = num.x;
  res.y = -num.y;
  return res;
}

__device__ inline void operator+=(double2 &lh, const double2 rh) {
  lh.x += rh.x;
  lh.y += rh.y;
}

__device__ inline void operator-=(double2 &lh, const double2 rh) {
  lh.x -= rh.x;
  lh.y -= rh.y;
}

__device__ inline double2 operator+(const double2 a, const double2 b) {
  double2 res;
  res.x = a.x + b.x;
  res.y = a.y + b.y;
  return res;
}

__device__ inline double2 operator-(const double2 a, const double2 b) {
  double2 res;
  res.x = a.x - b.x;
  res.y = a.y - b.y;
  return res;
}

__device__ inline double2 operator*(const double2 a, const double2 b) {
  double xx = a.x * b.x;
  double xy = a.x * b.y;
  double yx = a.y * b.x;
  double yy = a.y * b.y;

  double2 res;
  // asm volatile("fma.rn.f64 %0, %1, %2, %3;": "=d"(res.x) : "d"(a.x),
  // "d"(b.x), "d"(yy));
  res.x = xx - yy;
  res.y = xy + yx;
  return res;
}

__device__ inline double2 operator*(const double2 a, double b) {
  double2 res;
  res.x = a.x * b;
  res.y = a.y * b;
  return res;
}

__device__ inline void operator*=(double2 &a, const double2 b) {
  double tmp = a.x;
  a.x *= b.x;
  a.x -= a.y * b.y;
  a.y *= b.x;
  a.y += b.y * tmp;
}

__device__ inline double2 operator*(double a, double2 b) {
  double2 res;
  res.x = b.x * a;
  res.y = b.y * a;
  return res;
}

template <typename T> __global__ void print_debug_kernel(T *src, int N) {
  for (int i = 0; i < N; i++) {
    printf("%ul, ", src[i]);
  }
}

template <typename T> void print_debug(const char *name, T *src, int N) {
  printf("%s: ", name);
  cudaDeviceSynchronize();
  print_debug_kernel<<<1, 1>>>(src, N);
  cudaDeviceSynchronize();
  printf("\n");
}

#endif
