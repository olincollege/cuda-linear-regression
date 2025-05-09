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

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(models)

add_executable(regression regression.c)
target_link_libraries(regression PRIVATE linear_regression matrix cpu_matrix)

add_executable(benchmark benchmark.c)
target_link_libraries(benchmark PRIVATE linear_regression matrix cpu_matrix gpu_matrix timing)
