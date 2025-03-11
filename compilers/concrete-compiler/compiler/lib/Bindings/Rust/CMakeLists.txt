add_compile_options(-fexceptions)

add_custom_command(
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/lib.rs
  OUTPUT concrete/src/lib.rs.h concrete/src/lib.rs.cc
  COMMAND cxxbridge ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/lib.rs --header > concrete/src/lib.rs.h
  COMMAND cxxbridge ${CMAKE_CURRENT_SOURCE_DIR}/concrete/src/lib.rs > concrete/src/lib.rs.cc)

add_library(ConcreteRust SHARED concrete/src/lib.cpp concrete/src/lib.rs.cc)

target_include_directories(ConcreteRust PRIVATE concrete/src)

target_link_libraries(
  ConcreteRust
  PRIVATE ConcretelangSupport
          ConcretelangCommon
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
