add_library(linear_regression linear_regression.c linear_regression.h)
target_link_libraries(linear_regression PUBLIC matrix PRIVATE gpu_matrix cpu_matrix)

target_include_directories(matrix
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
