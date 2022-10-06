
#ifndef GPU_BOOTSTRAP_TWIDDLES_CUH
#define GPU_BOOTSTRAP_TWIDDLES_CUH

extern __constant__ short SW1[2048];
extern __constant__ short SW2[2048];

extern __constant__ double2 negTwids3[4];
extern __constant__ double2 negTwids4[8];
extern __constant__ double2 negTwids5[16];
extern __constant__ double2 negTwids6[32];
extern __constant__ double2 negTwids7[64];
extern __constant__ double2 negTwids8[128];
extern __constant__ double2 negTwids9[256];
extern __constant__ double2 negTwids10[512];
extern __constant__ double2 negTwids11[1024];
extern __device__ double2 negTwids12[2048];
extern __device__ double2 negTwids13[4096];

#endif
