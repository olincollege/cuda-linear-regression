# Some compilers warn that the return value of cr_assert gets converted from an
# int to a unsigned char:1. To avoid this error in those compilers, turn off
# conversion warnings for test code.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-conversion")

list(APPEND CMAKE_CTEST_ARGUMENTS --verbose)

add_executable(test_cpu_matrix test_cpu_matrix.c)
target_link_libraries(test_cpu_matrix
    PRIVATE cpu_matrix
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_cpu_matrix
    COMMAND test_cpu_matrix ${CRITERION_FLAGS}
)

add_executable(test_gpu_matrix test_gpu_matrix.c)
target_link_libraries(test_gpu_matrix
    PRIVATE gpu_matrix
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_gpu_matrix
    COMMAND test_gpu_matrix ${CRITERION_FLAGS}
)

add_executable(test_linear_regression test_linear_regression.c)
target_link_libraries(test_linear_regression
    PRIVATE linear_regression
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_linear_regression
    COMMAND test_linear_regression ${CRITERION_FLAGS}
)

add_executable(test_matrix test_matrix.c)
target_link_libraries(test_matrix
    PRIVATE matrix
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_matrix
    COMMAND test_matrix ${CRITERION_FLAGS}
)

add_executable(test_timing test_timing.c)
target_link_libraries(test_timing
    PRIVATE timing m
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_timing
    COMMAND test_timing ${CRITERION_FLAGS}
)
