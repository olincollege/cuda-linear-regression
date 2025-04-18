# Some compilers warn that the return value of cr_assert gets converted from an
# int to a unsigned char:1. To avoid this error in those compilers, turn off
# conversion warnings for test code.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-conversion")

list(APPEND CMAKE_CTEST_ARGUMENTS --verbose)

# If you need to test a library called foo, use lines like the following.
# add_executable(test_foo test_foo.c)
target_link_libraries(test_cpu_matrix
    PRIVATE cpu_matrix
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_cpu_matrix
    COMMAND test_cpu_matrix ${CRITERION_FLAGS}
)
target_link_libraries(test_gpu_matrix
    PRIVATE gpu_matrix
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_gpu_matrix
    COMMAND test_gpu_matrix ${CRITERION_FLAGS}
)
target_link_libraries(test_linear_regression
    PRIVATE linear_regression
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_linear_regression
    COMMAND test_linear_regression ${CRITERION_FLAGS}
)
target_link_libraries(test_matrix
    PRIVATE matrix
    PUBLIC ${CRITERION}
)
add_test(
    NAME test_matrix
    COMMAND test_matrix ${CRITERION_FLAGS}
)
