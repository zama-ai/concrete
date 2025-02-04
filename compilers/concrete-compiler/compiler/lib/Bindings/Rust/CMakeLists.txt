add_compile_options(-fexceptions)

add_custom_command(
        OUTPUT
        concrete-sys/src/lib.rs.h
        concrete-sys/src/lib.rs.cc
        COMMAND cxxbridge ${CMAKE_CURRENT_SOURCE_DIR}/concrete-sys/src/lib.rs --header > concrete-sys/src/lib.rs.h
        COMMAND cxxbridge ${CMAKE_CURRENT_SOURCE_DIR}/concrete-sys/src/lib.rs > concrete-sys/src/lib.rs.cc
)

add_library(
    ConcreteSys
    SHARED
    concrete-sys/src/lib.cpp
    concrete-sys/src/lib.rs.cc
)

target_include_directories(
    ConcreteSys
    PRIVATE
    concrete-sys/src
)

target_link_libraries(
    ConcreteSys
    PRIVATE
    ConcretelangSupport
    LLVMSupport
    _ConcreteSys_ConcretelangCommon_Sec
    _ConcreteSys_ConcretelangRuntime_Dfr
)
