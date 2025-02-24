include(FetchContent)

function(fetch_hpx_library)
    FetchContent_Declare(
        HPX
        GIT_REPOSITORY https://github.com/STEllAR-GROUP/hpx.git
        GIT_TAG 9c11915378e3c3901e49a5c3a67d6ee476fce3d1
        GIT_PROGRESS   TRUE
    )
    set(HPX_WITH_FETCH_ASIO ON CACHE BOOL INTERNAL)
    set(HPX_WITH_FETCH_HWLOC ON CACHE BOOL INTERNAL)
    set(HPX_WITH_FETCH_BOOST OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_MALLOC system CACHE STRING INTERNAL)
    set(HPX_WITH_EXAMPLES OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_TESTS OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_STATIC_LINKING OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_PKGCONFIG OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_MAX_CPU_COUNT "" CACHE STRING INTERNAL)
    set(HPX_WITH_CXX_STANDARD 17 CACHE STRING INTERNAL)
    set(HPX_WITH_COMPILER_WARNINGS OFF CACHE BOOL INTERNAL)
    unset(CMAKE_CXX_STANDARD CACHE)
    unset(CMAKE_CXX_STANDARD)
    remove_definitions("-Wall ")
    remove_definitions("-Werror ")
    remove_definitions("-Wfatal-errors")
    FetchContent_MakeAvailable(HPX)
    add_custom_target(
        HPXLibs
        ALL
        COMMAND ${CMAKE_COMMAND} -E copy
                ${hpx_BINARY_DIR}/lib/libhpx*
                ${CMAKE_BINARY_DIR}/lib
    )
    add_dependencies(HPXLibs
        HPX::hpx
        HPX::iostreams_component
        HPX::component
    )
endfunction()
