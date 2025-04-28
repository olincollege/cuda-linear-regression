#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include <time.h>

#include "timing.h"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
// Since we're testing, magic numbers are needed to configure specific cases

// Test that NANO_SEC_TO_SEC is correct
Test(timing_constants, nano_factor_correct) {
  cr_assert(ieee_ulp_eq(dbl, NANO_SEC_TO_SEC, 1e-9, 3),
            "Expected NANO_SEC_TO_SEC to be close to 1e-9 but got %f",
            NANO_SEC_TO_SEC);
}

// Test that get_current_time returns a positive number
Test(get_current_time, returns_non_negative) {
  double now = get_current_time();
  cr_assert(ge(dbl, now, 0.0), "Expected positive time but got %f", now);
}

// Test that time is increasing
Test(get_current_time, monotonic_non_decreasing) {
  double time_1 = get_current_time();
  double time_2 = get_current_time();
  cr_assert(ge(dbl, time_2, time_1),
            "Expected second time later than first time but got %f and %f",
            time_1, time_2);
}

// Test that sleeping increases the measured time
Test(get_current_time, measures_sleep_interval) {
  double start = get_current_time();

  struct timespec req;
  req.tv_sec = 0;
  req.tv_nsec = 100000000;  // 100 ms
  nanosleep(&req, NULL);

  double end = get_current_time();
  double elapsed = end - start;

  cr_assert(gt(dbl, elapsed, 0.05), "Expected elapsed time > 0.05s but got %f",
            elapsed);
  cr_assert(lt(dbl, elapsed, 0.2), "Expected elapsed time < 0.2s but got %f",
            elapsed);
}

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
