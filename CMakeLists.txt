cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(concrete_cuda LANGUAGES CXX CUDA)

include(CTest)

# See if the minimum CUDA version is available. If not, only enable documentation building.
set(MINIMUM_SUPPORTED_CUDA_VERSION 10.0)
include(CheckLanguage)
# See if CUDA is available
check_language(CUDA)
# If so, enable CUDA to check the version.
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif ()
# If CUDA is not available, or the minimum version is too low do not build
if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Cuda compiler not found.")
endif()

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${MINIMUM_SUPPORTED_CUDA_VERSION})
    message(FATAL_ERROR "CUDA ${MINIMUM_SUPPORTED_CUDA_VERSION} or greater is required for compilation.")
endif()
#Get CUDA compute capability
set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_script) # No suffix required
set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/check_cuda.cu)
execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
execute_process(COMMAND ${OUTPUTFILE}
        RESULT_VARIABLE CUDA_RETURN_CODE
        OUTPUT_VARIABLE ARCH)

if (${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
else ()
    set(CUDA_SUCCESS "FALSE")
endif ()

if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: ${ARCH}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

    set(CUDA_NVCC_FLAGS "${ARCH}")
    #add_definitions(-DGPU) #You may not require this

else ()
    message(WARNING ${ARCH})
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -g")
if (NOT CUDA_NVCC_FLAGS)
    set(CUDA_NVCC_FLAGS -arch=sm_70)
endif ()

# in production, should use -arch=sm_70
# --ptxas-options=-v to see register spills
# -lineinfo for better debugging
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER} -O3 ${CUDA_NVCC_FLAGS} \
  -std=c++17 --no-exceptions  --expt-relaxed-constexpr -rdc=true --use_fast_math -Xcompiler -fPIC")

set(INCLUDE_DIR include)

add_subdirectory(src)
add_subdirectory(parameters)
target_include_directories(concrete_cuda PRIVATE ${INCLUDE_DIR})

# This is required for rust cargo build
install(TARGETS concrete_cuda DESTINATION .)
install(TARGETS concrete_cuda DESTINATION lib)
install(TARGETS cuda_parameters DESTINATION .)
install(TARGETS cuda_parameters DESTINATION lib)

# Define a function to add a lint target.
find_file(CPPLINT NAMES cpplint cpplint.exe)
if (CPPLINT)
    # Add a custom target to lint all child projects. Dependencies are specified in child projects.
    add_custom_target(all_lint)
    # Don't trigger this target on ALL_BUILD or Visual Studio 'Rebuild Solution'
    set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_ALL TRUE)
    # set_target_properties(all_lint PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
endif ()
