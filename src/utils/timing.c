#include <time.h>

const double NANO_SEC_TO_SEC = 1e-9;

double get_current_time() {
  struct timespec t_spec;
  clock_gettime(CLOCK_MONOTONIC, &t_spec);
  return (double)t_spec.tv_sec + (double)t_spec.tv_nsec * NANO_SEC_TO_SEC;
}
