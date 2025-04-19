add_library(matrix matrix.cu matrix.hpp)
set_target_properties(matrix PROPERTIES LINKER_LANGUAGE CUDA)

add_library(timing timing.cpp timing.hpp)

target_include_directories(matrix
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(timing
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
