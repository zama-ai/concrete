set(SOURCES 
        ${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}/boolean_gates.h
        ${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}/bootstrap.h
        ${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}/keyswitch.h
        ${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}/linear_algebra.h
        ${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}/vertical_packing.h
        ${CMAKE_SOURCE_DIR}/${INCLUDE_DIR}/circuit_bootstrap.h)
file(GLOB SOURCES
     "*.cu"
     "*.h"
     "fft/*.cu")
add_library(concrete_cuda STATIC ${SOURCES})
set_target_properties(concrete_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(concrete_cuda PUBLIC cudart)
target_include_directories(concrete_cuda PRIVATE .)
