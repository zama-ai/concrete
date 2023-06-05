#ifndef LOG_H
#define LOG_H

#include <gmp.h>

#define LOG_LEVEL_DEBUG 3
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_WARN  1
#define LOG_LEVEL_ERROR 0

#define LOG_LEVEL LOG_LEVEL_WARN
#define LOG_OUTSTREAM stderr

/* vt100 escapes codes for colors. Not portable, could be changed later */
#define COLOR_RED    fprintf(LOG_OUTSTREAM, "\033[31m")
#define COLOR_WHITE  fprintf(LOG_OUTSTREAM, "\033[37m")
#define COLOR_GREEN  fprintf(LOG_OUTSTREAM, "\033[32m")
#define COLOR_YELLOW fprintf(LOG_OUTSTREAM, "\033[33m")
#define COLOR_OFF    fprintf(LOG_OUTSTREAM, "\033[0m")

/* Level 3 - Debugging */
#if LOG_LEVEL >= LOG_LEVEL_DEBUG
#   define log_debug(fmt, ...) COLOR_WHITE, log_(fmt, ##__VA_ARGS__), COLOR_OFF

#else
#   define log_debug(fmt, ...)
#endif
/* Level 2 - Info */
#if LOG_LEVEL >= LOG_LEVEL_INFO
#   define log_info(fmt, ...) log_(fmt, ##__VA_ARGS__)
#else
#   define log_info(fmt, ...)
#endif
/* Level 1  - Warnings */
#if LOG_LEVEL >= LOG_LEVEL_WARN
#   define log_warn(fmt, ...) COLOR_YELLOW, log_(fmt, ##__VA_ARGS__), COLOR_OFF
#else
#   define log_info(fmt, ...)
#endif
/* Level 0 - Errors, Fatal*/
#if LOG_LEVEL >= LOG_LEVEL_ERROR
#   define log_error(fmt, ...) COLOR_RED, log_(fmt, ##__VA_ARGS__), COLOR_OFF
#   define log_fatal(fmt, ...) COLOR_RED, log_(fmt, ##__VA_ARGS__), COLOR_OFF
#else
#   define log_error(fmt, ...)
#   define log_fatal(fmt, ...)
#endif

/* Log output is of form 'program name : file (line number) : log message' */
#define log_(fmt, ...) gmp_fprintf(LOG_OUTSTREAM, "  %-18s (l:%3d): " fmt "\n",\
                               __FILE__, __LINE__, ##__VA_ARGS__)

#endif
