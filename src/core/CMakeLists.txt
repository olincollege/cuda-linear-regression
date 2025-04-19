add_library(cpu_matrix cpu_matrix.cpp cpu_matrix.hpp)
target_link_libraries(cpu_matrix PUBLIC matrix)

add_library(gpu_matrix gpu_matrix.cu gpu_matrix.hpp)
target_link_libraries(gpu_matrix PUBLIC matrix)
set_target_properties(gpu_matrix PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(cpu_matrix
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(gpu_matrix
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
