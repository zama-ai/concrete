#ifndef PAPI_H
#define PAPI_H

#ifdef __cplusplus
extern "C" {
#endif

static const char *papi_events[] = {"PAPI_TOT_INS", /* 0 */
                                    "PAPI_TOT_CYC", /* 1 */
                                    "PAPI_LD_INS",  /* 2 */
                                    "PAPI_L1_DCM",  /* 3 */
                                    "PAPI_L2_DCM",  /* 4 */
                                    "PAPI_L3_TCM",  /* 5 */
                                    "PAPI_BR_MSP",  /* 6 */
                                    "PAPI_BR_INS" /* 7 */};

enum PAPI_REPORT_FLAG {
  PAPI_REPORT_PLAIN,
  PAPI_REPORT_RATIO,
  PAPI_REPORT_RATIO_PERCENT
};

static const char *papi_report_labels[] = {"CPI",  "LDpi", "L1pl",
                                           "L2pl", "L3pl", "BRMR"};
static const int papi_reports[] = {PAPI_REPORT_RATIO,         1, 0,

                                   PAPI_REPORT_RATIO_PERCENT, 2, 0,

                                   PAPI_REPORT_RATIO_PERCENT, 3, 2,

                                   PAPI_REPORT_RATIO_PERCENT, 4, 2,

                                   PAPI_REPORT_RATIO_PERCENT, 5, 2,

                                   PAPI_REPORT_RATIO_PERCENT, 6, 7};

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#define NUM_PAPI_EVENTS ARRAY_SIZE(papi_events)
#define NUM_PAPI_REPORTS ARRAY_SIZE(papi_reports)

int setup_papi(void);
int read_papi_values(long long values[NUM_PAPI_EVENTS]);

#ifdef __cplusplus
}
#endif

#endif
