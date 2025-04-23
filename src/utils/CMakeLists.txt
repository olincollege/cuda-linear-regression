add_library(matrix matrix.cu matrix.h)
set_target_properties(matrix PROPERTIES LINKER_LANGUAGE CUDA)

add_library(timing timing.c timing.h)

target_include_directories(matrix
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(timing
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
