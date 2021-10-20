file(GLOB SOURCES
        "parameters.cpp")
add_library(cuda_parameters STATIC ${SOURCES})
set_target_properties(cuda_parameters PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
