// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_TIME_UTIL_H
#define CONCRETELANG_DFR_TIME_UTIL_H

#if CONCRETELANG_TIMING_ENABLED

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "concretelang/Runtime/DFRuntime.hpp"

#define TIME_UTIL_CLOCK CLOCK_MONOTONIC

namespace mlir {
namespace concretelang {
namespace time_util {

extern bool timing_enabled;
extern struct timespec timestamp;

} // namespace time_util
} // namespace concretelang
} // namespace mlir

static inline int timespec_diff(struct timespec *, const struct timespec *,
                                const struct timespec *);

#define CONCRETELANG_ENABLE_TIMING()                                           \
  do {                                                                         \
    assert(clock_gettime(TIME_UTIL_CLOCK,                                      \
                         &mlir::concretelang::time_util::timestamp) == 0);     \
    char *env = getenv("CONCRETE_TIMING_ENABLED");                             \
    if (env != nullptr)                                                        \
      if (!strncmp(env, "True", 4) || !strncmp(env, "true", 4) ||              \
          !strncmp(env, "ON", 2) || !strncmp(env, "on", 2) ||                  \
          !strncmp(env, "1", 1))                                               \
        mlir::concretelang::time_util::timing_enabled = true;                  \
  } while (0)

#define BEGIN_TIME(p)                                                          \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      assert(clock_gettime(TIME_UTIL_CLOCK, (p)) == 0);                        \
    }                                                                          \
  } while (0)

#if CONCRETELANG_DATAFLOW_EXECUTION_ENABLED
#define END_TIME(p, m)                                                         \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      struct timespec _end_time_tv;                                            \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_end_time_tv) == 0);              \
      assert(timespec_diff((p), &_end_time_tv, (p)) == 0);                     \
      struct timespec _timestamp_tv;                                           \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_timestamp_tv) == 0);             \
      assert(timespec_diff(&_timestamp_tv, &_timestamp_tv,                     \
                           &mlir::concretelang::time_util::timestamp) == 0);   \
      std::cout << "[Timing logs][" << time_in_seconds(&_timestamp_tv)         \
                << "] -\t";                                                    \
      std::cout << "[NODE \t" << _dfr_debug_get_node_id() << "] \t" << (m)     \
                << " time : \t" << time_in_seconds((p)) << " seconds.\n"       \
                << std::flush;                                                 \
    }                                                                          \
  } while (0)
#define END_TIME_C(p, m, c)                                                    \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      struct timespec _end_time_tv;                                            \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_end_time_tv) == 0);              \
      assert(timespec_diff((p), &_end_time_tv, (p)) == 0);                     \
      struct timespec _timestamp_tv;                                           \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_timestamp_tv) == 0);             \
      assert(timespec_diff(&_timestamp_tv, &_timestamp_tv,                     \
                           &mlir::concretelang::time_util::timestamp) == 0);   \
      std::cout << "[Timing logs][" << time_in_seconds(&_timestamp_tv)         \
                << "] -\t";                                                    \
      std::cout << "[NODE \t" << _dfr_debug_get_node_id() << "] \t" << (m)     \
                << " [" << (c) << "] time : \t" << time_in_seconds((p))        \
                << " seconds.\n"                                               \
                << std::flush;                                                 \
    }                                                                          \
  } while (0)
#define END_TIME_C_ACC(p, m, c, acc)                                           \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      struct timespec _end_time_tv;                                            \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_end_time_tv) == 0);              \
      assert(timespec_diff((p), &_end_time_tv, (p)) == 0);                     \
      timespec_acc((acc), (p), (acc));                                         \
      struct timespec _timestamp_tv;                                           \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_timestamp_tv) == 0);             \
      assert(timespec_diff(&_timestamp_tv, &_timestamp_tv,                     \
                           &mlir::concretelang::time_util::timestamp) == 0);   \
      std::cout << "[Timing logs][" << time_in_seconds(&_timestamp_tv)         \
                << "] -\t";                                                    \
      std::cout << "[NODE \t" << _dfr_debug_get_node_id() << "] \t" << (m)     \
                << " [" << (c) << "] time : \t" << time_in_seconds((p))        \
                << " (total : " << time_in_seconds((acc)) << " )"              \
                << " seconds.\n"                                               \
                << std::flush;                                                 \
    }                                                                          \
  } while (0)
#else
#define END_TIME(p, m)                                                         \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      struct timespec _end_time_tv;                                            \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_end_time_tv) == 0);              \
      assert(timespec_diff((p), &_end_time_tv, (p)) == 0);                     \
      struct timespec _timestamp_tv;                                           \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_timestamp_tv) == 0);             \
      assert(timespec_diff(&_timestamp_tv, &_timestamp_tv,                     \
                           &mlir::concretelang::time_util::timestamp) == 0);   \
      std::cout << "[Timing logs][" << time_in_seconds(&_timestamp_tv)         \
                << "] -\t";                                                    \
      std::cout << (m) << " time : \t" << time_in_seconds((p))                 \
                << " seconds.\n"                                               \
                << std::flush;                                                 \
    }                                                                          \
  } while (0)
#define END_TIME_C(p, m, c)                                                    \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      struct timespec _end_time_tv;                                            \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_end_time_tv) == 0);              \
      assert(timespec_diff((p), &_end_time_tv, (p)) == 0);                     \
      struct timespec _timestamp_tv;                                           \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_timestamp_tv) == 0);             \
      assert(timespec_diff(&_timestamp_tv, &_timestamp_tv,                     \
                           &mlir::concretelang::time_util::timestamp) == 0);   \
      std::cout << "[Timing logs][" << time_in_seconds(&_timestamp_tv)         \
                << "] -\t";                                                    \
      std::cout << (m) << " [" << (c) << "] time : \t" << time_in_seconds((p)) \
                << " seconds.\n"                                               \
                << std::flush;                                                 \
    }                                                                          \
  } while (0)
#define END_TIME_C_ACC(p, m, c, acc)                                           \
  do {                                                                         \
    if (mlir::concretelang::time_util::timing_enabled) {                       \
      struct timespec _end_time_tv;                                            \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_end_time_tv) == 0);              \
      assert(timespec_diff((p), &_end_time_tv, (p)) == 0);                     \
      timespec_acc((acc), (p), (acc));                                         \
      struct timespec _timestamp_tv;                                           \
      assert(clock_gettime(TIME_UTIL_CLOCK, &_timestamp_tv) == 0);             \
      assert(timespec_diff(&_timestamp_tv, &_timestamp_tv,                     \
                           &mlir::concretelang::time_util::timestamp) == 0);   \
      std::cout << "[Timing logs][" << time_in_seconds(&_timestamp_tv)         \
                << "] -\t";                                                    \
      std::cout << (m) << " [" << (c) << "] time : \t" << time_in_seconds((p)) \
                << " (total : " << time_in_seconds((acc)) << " )"              \
                << " seconds.\n"                                               \
                << std::flush;                                                 \
    }                                                                          \
  } while (0)
#endif

static inline double get_thread_cpu_time(void) {
  struct timespec _tv;
  double _t;

  assert(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &_tv) == 0);
  _t = _tv.tv_sec;
  _t += _tv.tv_nsec * 1e-9;
  return _t;
}

static inline double time_in_seconds(struct timespec *_tv) {
  double _t;
  _t = _tv->tv_sec;
  _t += _tv->tv_nsec * 1e-9;
  return _t;
}

static inline int timespec_diff(struct timespec *_result,
                                const struct timespec *_px,
                                const struct timespec *_py) {
  struct timespec _x, _y;

  _x = *_px;
  _y = *_py;

  /* Perform the carry for the later subtraction by updating y. */
  if (_x.tv_nsec < _y.tv_nsec) {
    long _ns = (_y.tv_nsec - _x.tv_nsec) / 1000000000L + 1;
    _y.tv_nsec -= 1000000000L * _ns;
    _y.tv_sec += _ns;
  }
  if (_x.tv_nsec - _y.tv_nsec > 1000000000L) {
    long _ns = (_x.tv_nsec - _y.tv_nsec) / 1000000000L;
    _y.tv_nsec += 1000000000L * _ns;
    _y.tv_sec -= _ns;
  }

  /* Compute the time remaining to wait. tv_nsec is certainly
     positive. */
  _result->tv_sec = _x.tv_sec - _y.tv_sec;
  _result->tv_nsec = _x.tv_nsec - _y.tv_nsec;

  /* Return 1 if result is negative. */
  return _x.tv_sec < _y.tv_sec;
}

static inline void timespec_acc(struct timespec *_result,
                                const struct timespec *_px,
                                const struct timespec *_py) {
  struct timespec _x, _y;
  _x = *_px;
  _y = *_py;
  _result->tv_sec = _x.tv_sec + _y.tv_sec;
  _result->tv_nsec = _x.tv_nsec + _y.tv_nsec;
}

#else // CONCRETELANG_TIMING_ENABLED

#define CONCRETELANG_ENABLE_TIMING()
#define BEGIN_TIME(p)
#define END_TIME(p, m)
#define END_TIME_C(p, m, c)
#define END_TIME_C_ACC(p, m, c, acc)

#endif // CONCRETELANG_TIMING_ENABLED
#endif
