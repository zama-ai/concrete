#include "profiling.h"
#include <papi.h>
#include <stdio.h>
#include <string.h>

static int papi_event_set = PAPI_NULL;
static long long papi_counter_values[NUM_PAPI_EVENTS] = {0};

int setup_papi(void) {
  static int initialized = 0;

  if (initialized)
    return 0;

  int retval;

  retval = PAPI_library_init(PAPI_VER_CURRENT);

  if (retval != PAPI_VER_CURRENT && retval > 0) {
    fprintf(stderr, "PAPI library version mismatch!\n");
    return 1;
  }

  if (retval < 0) {
    fprintf(stderr, "Could not init PAPI library: %s (%d) %d!\n",
            PAPI_strerror(retval), retval, PAPI_VER_CURRENT);
    return 1;
  }

  if (PAPI_is_initialized() != PAPI_LOW_LEVEL_INITED) {
    fprintf(stderr, "Could not init PAPI library (low-level part)!\n");
    return 1;
  }

#ifdef PAPI_MULTIPLEX
  if (PAPI_multiplex_init() != PAPI_OK) {
    fprintf(stderr, "Could not init multiplexing!\n");
    return 1;
  }
#endif

  int err;

  /* Create event set */
  if ((err = PAPI_create_eventset(&papi_event_set)) != PAPI_OK) {
    fprintf(stderr, "Could not create event set: %s!\n", PAPI_strerror(err));
    return 1;
  }

#ifdef WS_PAPI_MULTIPLEX
  /* Enable multiplexing for event set */
  if ((err = PAPI_set_multiplex(papi_event_set)) != PAPI_OK) {
    fprintf(stderr, "Could not enable multiplexing for event set: %s!\n",
            PAPI_strerror(err));
    return 1;
  }
#endif

  for (size_t i = 0; i < NUM_PAPI_EVENTS; i++) {
    if ((err = PAPI_add_named_event(papi_event_set, papi_events[i])) !=
        PAPI_OK) {
      fprintf(stderr, "Could not add event \"%s\" to event set: %s!\n",
              papi_events[i], PAPI_strerror(err));
      return 1;
    }
  }

  if (PAPI_start(papi_event_set) != PAPI_OK) {
    fputs("Could not start event set", stderr);
    return 1;
  }

  initialized = 1;

  return 0;
}

int read_papi_values(long long values[NUM_PAPI_EVENTS]) {
  long long new_values[NUM_PAPI_EVENTS] = {0};

  if (PAPI_accum(papi_event_set, new_values) != PAPI_OK) {
    fputs("Could not read counters", stderr);
    return 1;
  }

  /* /\* for(size_t i = 0; i < NUM_PAPI_EVENTS; i++) { *\/ */
  /* printf("Curr val: %s: %llu\n", papi_events[4], new_values[4]); */
  /* printf("Curr val: %s: %llu\n", papi_events[5], new_values[5]); */
  /* printf("Curr rate: %.2f%%\n", */
  /*        100.0 * ((double)new_values[4] / (double)new_values[5])); */
  /* /\* } *\/ */

  for (size_t i = 0; i < NUM_PAPI_EVENTS; i++) {
    papi_counter_values[i] += new_values[i];
    values[i] = papi_counter_values[i];
  }

  //  memcpy(values, papi_counter_values, sizeof(papi_counter_values));

  return 0;
}
