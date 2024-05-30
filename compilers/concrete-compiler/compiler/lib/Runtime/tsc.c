#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdint.h>
#include <inttypes.h>

uint64_t get_current_nanoseconds(void) {
  struct timespec ts;

  clock_gettime(CLOCK_REALTIME, &ts);
  return 1000000000 * (uint64_t)ts.tv_sec + ts.tv_nsec;
}
