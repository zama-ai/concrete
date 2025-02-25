include(FetchContent)

function(fetch_hpx_library)
    FetchContent_Declare(
        HPX
        GIT_REPOSITORY https://github.com/STEllAR-GROUP/hpx.git
        GIT_TAG stable # release 1.10 + fixes for compilation with fopenmp with gcc and clang :,)
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
    )
    set(HPX_WITH_FETCH_ASIO ON CACHE BOOL INTERNAL)
    set(HPX_WITH_FETCH_HWLOC ON CACHE BOOL INTERNAL)
    set(HPX_WITH_FETCH_BOOST OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_MALLOC system CACHE STRING INTERNAL)
    set(HPX_WITH_EXAMPLES OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_TESTS OFF CACHE BOOL INTERNAL)
    set(HPX_WITH_STATIC_LINKING ON CACHE BOOL INTERNAL)
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
endfunction()
