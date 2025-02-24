add_compile_options(-fexceptions)

add_custom_command(
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/ffi.rs ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/ffi.h
  OUTPUT concrete/src/ffi.rs.cc ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/cxx.h
  COMMAND cxxbridge ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/ffi.rs > concrete/src/ffi.rs.cc
  COMMAND cxxbridge --header > ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/cxx.h)

add_library(ConcreteRust SHARED concrete/src/ffi.h concrete/src/ffi.rs.cc)

target_include_directories(ConcreteRust PRIVATE concrete/src)

if(LINUX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")
endif()

target_link_libraries(
  ConcreteRust
  PRIVATE ConcretelangSupport
          ConcretelangCommon
          ConcretelangClientLib
          ConcretelangServerLib
          ConcretelangRuntimeStatic
          LLVMSupport
          capnp
          capnp-json
          kj)

if(APPLE)
  find_library(SECURITY_FRAMEWORK Security)
  if(NOT SECURITY_FRAMEWORK)
    message(FATAL_ERROR "Security framework not found")
  endif()
  target_link_libraries(ConcreteRust LINK_PUBLIC ${SECURITY_FRAMEWORK})
endif()
