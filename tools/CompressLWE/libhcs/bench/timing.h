#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define CLOCK_TYPE CLOCK_REALTIME

static inline int cmptime(const void *a, const void *b)
{
    struct timespec ca = *(struct timespec*)a;
    struct timespec cb = *(struct timespec*)b;

    if (ca.tv_sec != cb.tv_sec) {
        return ca.tv_sec > cb.tv_sec ? 1 : 0;
    }
    else {
        return ca.tv_nsec >= cb.tv_nsec ? 1 : 0;
    }
}

#define TIME_CODE(msg, tests, code) \
    do {\
        printf("Timing: %s\n", msg);\
        struct timespec all_times[tests];  \
        for (int __i = 0; __i < (tests); ++__i) { \
            struct timespec _ts_start, _ts_end, _ts_calc;\
            clock_gettime(CLOCK_TYPE, &_ts_start);\
            code\
            clock_gettime(CLOCK_TYPE, &_ts_end);\
            if (_ts_end.tv_nsec - _ts_start.tv_nsec < 0) {\
                _ts_calc.tv_sec = _ts_end.tv_sec - _ts_start.tv_sec - 1;\
                _ts_calc.tv_nsec = 1e9 + _ts_end.tv_nsec - _ts_start.tv_nsec;\
            }\
            else {\
                _ts_calc.tv_sec = _ts_end.tv_sec - _ts_start.tv_sec;\
                _ts_calc.tv_nsec = _ts_end.tv_nsec - _ts_start.tv_nsec;\
            }\
            all_times[__i] = _ts_calc;  \
            printf("%d:\n\t%lus %juns\n\n", __i, _ts_calc.tv_sec, _ts_calc.tv_nsec);\
        }   \
        for (int __i = 0; __i < (tests); ++__i) {\
            printf("%d:\n\t%lus %juns\n\n", __i, all_times[__i].tv_sec, all_times[__i].tv_nsec);\
        }\
    } while (0)

#define MTIME_CODE(code)\
    ({\
        struct timespec _ts_start, _ts_end, _ts_calc;\
        clock_gettime(CLOCK_TYPE, &_ts_start);\
        code\
        clock_gettime(CLOCK_TYPE, &_ts_end);\
        if (_ts_end.tv_nsec - _ts_start.tv_nsec < 0) {\
            _ts_calc.tv_sec = _ts_end.tv_sec - _ts_start.tv_sec - 1;\
            _ts_calc.tv_nsec = 1e9 + _ts_end.tv_nsec - _ts_start.tv_nsec;\
        }\
        else {\
            _ts_calc.tv_sec = _ts_end.tv_sec - _ts_start.tv_sec;\
            _ts_calc.tv_nsec = _ts_end.tv_nsec - _ts_start.tv_nsec;\
        }\
        _ts_calc; \
    })
