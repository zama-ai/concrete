#ifndef SEGMENT_H
#define SEGMENT_H

#include "profiling.h"
#include "tsc.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#define compiler_fence() std::atomic_thread_fence(std::memory_order_release)
#else
#define compiler_fence() atomic_thread_fence(memory_order_release)
#endif

#ifdef NO_PROFILING

#define SEGMENT_DECLARE(name, pparent)
#define SEGMENT_DECLARE_ROOT(name)
#define SEGMENT_START(name)
#define SEGMENT_END(name)
#define SEGMENT_START_ROOT(name)
#define SEGMENT_END_ROOT(name)
#define SEGMENT_STATS_INDENT(...)
#define SEGMENT_STATS(root)
#define SEGMENT_STATS_TC(...)

#else

struct segment {
  const char *descr;
  int init;
  uint64_t start;
  uint64_t end;
  uint64_t total;
  uint64_t tripcount;
  struct segment *next;
  struct segment *parent;
  struct segment *last_segment;
  struct segment *all_segments;

#ifndef NO_PAPI_PROFILING
  long long papi_values_start[NUM_PAPI_EVENTS];
  long long papi_values_end[NUM_PAPI_EVENTS];
  long long papi_values_accu[NUM_PAPI_EVENTS];
#endif
};

#define SEGMENT_START_ROOT(name) __SEGMENT_START_INTERNAL(name)
#define SEGMENT_END_ROOT(name) __SEGMENT_END_INTERNAL(name)

#ifdef ROOT_PROFILING
#define SEGMENT_DECLARE(name, pparent)
#define SEGMENT_START(name)
#define SEGMENT_END(name)
#else
#define SEGMENT_DECLARE(name, pparent) __SEGMENT_DECLARE_INTERNAL(name, pparent)
#define SEGMENT_START(name) __SEGMENT_START_INTERNAL(name)
#define SEGMENT_END(name) __SEGMENT_END_INTERNAL(name)
#endif

#ifdef NO_PAPI_PROFILING
#define __SEGMENT_INIT_PAPI_INTERNAL(name)
#define __SEGMENT_START_PAPI_INTERNAL(name)
#define __SEGMENT_END_PAPI_INTERNAL(name)
#else
#define __SEGMENT_INIT_PAPI_INTERNAL(name)                                     \
  memset(name.papi_values_start, 0, sizeof(name.papi_values_start));           \
  memset(name.papi_values_end, 0, sizeof(name.papi_values_end));               \
  memset(name.papi_values_accu, 0, sizeof(name.papi_values_accu));

#define __SEGMENT_START_PAPI_INTERNAL(name)                                    \
  if (read_papi_values(name.papi_values_start) != 0)                           \
    exit(1);

#define __SEGMENT_END_PAPI_INTERNAL(name)                                      \
  if (read_papi_values(name.papi_values_end) != 0)                             \
    exit(1);                                                                   \
                                                                               \
  for (size_t i = 0; i < NUM_PAPI_EVENTS; i++) {                               \
    name.papi_values_accu[i] +=                                                \
        name.papi_values_end[i] - name.papi_values_start[i];                   \
  }

#endif

#define __SEGMENT_DECLARE_INTERNAL(name, pparent)                              \
  static struct segment name = {0};                                            \
                                                                               \
  if (!name.init) {                                                            \
    name.init = 1;                                                             \
    struct segment *__pparent = pparent;                                       \
                                                                               \
    if (__pparent) {                                                           \
      if (!__pparent->all_segments)                                            \
        __pparent->all_segments = &name;                                       \
                                                                               \
      if (__pparent->last_segment)                                             \
        __pparent->last_segment->next = &name;                                 \
                                                                               \
      __pparent->last_segment = &name;                                         \
    }                                                                          \
                                                                               \
    name.descr = #name;                                                        \
    name.total = 0;                                                            \
    name.tripcount = 0;                                                        \
    name.next = NULL;                                                          \
    name.parent = pparent;                                                     \
    name.all_segments = NULL;                                                  \
    name.last_segment = NULL;                                                  \
    __SEGMENT_INIT_PAPI_INTERNAL(name);                                        \
  }

#define SEGMENT_DECLARE_ROOT(name) __SEGMENT_DECLARE_INTERNAL(name, NULL);

#define __SEGMENT_START_INTERNAL(name)                                         \
  compiler_fence();                                                            \
  name.start = get_current_nanoseconds();                                      \
  __SEGMENT_START_PAPI_INTERNAL(name);

#define __SEGMENT_END_INTERNAL(name)                                           \
  compiler_fence();                                                            \
  name.end = get_current_nanoseconds();                                        \
  name.total += name.end - name.start;                                         \
  name.tripcount++;                                                            \
                                                                               \
  __SEGMENT_END_PAPI_INTERNAL(name);

static inline void SEGMENT_STATS_INDENT(struct segment *root,
                                        struct segment *parent,
                                        struct segment *s, int indent) {
  printf("%*s", 2 * indent, "");

  printf("Segment %*s [%p]: %03.2f%%R [%03.2f%%T %" PRIu64 "ns, %" PRIu64
         "it, %.2fns/it] ",
         50 - 2 * indent, s->descr, (void *)root,
         100.0 * ((double)s->total / (double)parent->total),
         100.0 * ((double)s->total / (double)root->total), s->total,
         s->tripcount, s->total / (double)s->tripcount);

#ifndef NO_PAPI_PROFILING
  size_t label = 0;
  size_t i = 0;
  while (i < NUM_PAPI_REPORTS) {
    printf("%s%s: ", (i == 0) ? "" : ", ", papi_report_labels[label]);
    label++;

    if (papi_reports[i] == PAPI_REPORT_PLAIN) {
      printf("%llu", s->papi_values_accu[papi_reports[i + 1]]);
      i += 2;
    } else if (papi_reports[i] == PAPI_REPORT_RATIO) {
      printf("%.5f", ((double)s->papi_values_accu[papi_reports[i + 1]] /
                      (double)s->papi_values_accu[papi_reports[i + 2]]));

      i += 3;
    } else if (papi_reports[i] == PAPI_REPORT_RATIO_PERCENT) {
      printf("%.2f%%",
             100.0 * ((double)s->papi_values_accu[papi_reports[i + 1]] /
                      (double)s->papi_values_accu[papi_reports[i + 2]]));

      i += 3;
    }
  }
#endif

  puts("");

  for (struct segment *child = s->all_segments; child; child = child->next) {
    SEGMENT_STATS_INDENT(root, s, child, indent + 1);
  }
}

static inline void SEGMENT_STATS(struct segment *root) {
  SEGMENT_STATS_INDENT(root, root, root, 0);
}

static inline void SEGMENT_STATS_TC(struct segment *root, uint64_t tc) {
  if (root->tripcount % tc == 0) {
    SEGMENT_STATS_INDENT(root, root, root, 0);
  }
}

#endif

#endif // SEGMENT_H
