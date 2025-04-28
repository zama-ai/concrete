add_compile_options(-fexceptions)

add_custom_command(
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ffi.rs ${CMAKE_CURRENT_SOURCE_DIR}/ffi.h
  OUTPUT ffi.rs.cc ${CMAKE_CURRENT_SOURCE_DIR}/cxx.h
  COMMAND cxxbridge ${CMAKE_CURRENT_SOURCE_DIR}/ffi.rs > ffi.rs.cc
  COMMAND cxxbridge --header > ${CMAKE_CURRENT_SOURCE_DIR}/cxx.h)

add_library(ConcreteRust SHARED ffi.rs.cc)

if(LINUX)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")
endif()

target_link_libraries(
  ConcreteRust
  PRIVATE ConcretelangSupport
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
