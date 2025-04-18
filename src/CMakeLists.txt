# If you have a program that's just a .c file and it has a main method, define
# an executable. If it has both a .c and .h file, make it a library. Also, link
# libraries appropriately here. Targets should be added in dependency order - if
# executable bar needs library foo, then you should do
#   add_library(foo foo.c foo.h)
#   add_executable(bar bar.c)
#   target_link_libraries(bar PRIVATE foo)
# or similar. The PRIVATE/INTERFACE/PUBLIC keyword will depend on whether the
# library is used only in function bodies (PRIVATE), only in function
# signatures/types (INTERFACE), or both (PUBLIC).
add_library(cpu_matrix core/cpu_matrix.cpp core/cpu_matrix.hpp)
add_library(gpu_matrix core/gpu_matrix.cu core/gpu_matrix.hpp)

add_library(linear_regression models/linear_regression.cpp models/linear_regression.hpp)

add_library(matrix utils/matrix.cu utils/matrix.hpp)
add_library(timing utils/timing.cpp utils/timing.hpp)

target_link_libraries(matrix PUBLIC cpu_matrix)
target_link_libraries(matrix PUBLIC gpu_matrix)
target_link_libraries(cpu_matrix PUBLIC linear_regression)

