cmake_minimum_required(VERSION 3.17)

project(concrete-protocol CXX)
include(ExternalProject)

add_compile_options(-fPIC)

set(PROTOBUF_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf_bin_dir)
set(PROTOBUF_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf_src_dir)
set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_SRC_DIR}/src)
file(MAKE_DIRECTORY "${PROTOBUF_BIN_DIR}")
set(PROTOBUF_LIBRARY ${PROTOBUF_BIN_DIR}/libprotobuf.a)
set(PROTOBUF_PROTOC ${PROTOBUF_BIN_DIR}/protoc)

ExternalProject_Add(
  protobuf_repo
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v21.12
  GIT_SUBMODULES_RECURSE ON
  GIT_PROGRESS   TRUE
  BUILD_ALWAYS 1
  INSTALL_COMMAND cp ${PROTOBUF_LIBRARY} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  BINARY_DIR ${PROTOBUF_BIN_DIR}
  SOURCE_DIR ${PROTOBUF_SRC_DIR}
  CMAKE_ARGS -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  BUILD_BYPRODUCTS ${PROTOBUF_LIBRARY} ${PROTOBUF_PROTOC}
)

add_library(protobuf STATIC IMPORTED)
set_property(TARGET protobuf PROPERTY IMPORTED_LOCATION ${PROTOBUF_LIBRARY})
add_dependencies(protobuf protobuf_repo)

add_executable(protoc IMPORTED)
set_property(TARGET protoc PROPERTY IMPORTED_LOCATION ${PROTOBUF_PROTOC})
add_dependencies(protoc protobuf_repo)

get_filename_component(CONCRETE_PROTOCOL_FILE "src/concrete-protocol.proto" ABSOLUTE)
get_filename_component(CONCRETE_PROTOCOL_FOLDER "${CONCRETE_PROTOCOL_FILE}" DIRECTORY)
set(GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
file(MAKE_DIRECTORY "${GENERATED_DIRECTORY}")
get_filename_component(GENERATED_CPP "${GENERATED_DIRECTORY}/concrete-protocol.pb.cc" ABSOLUTE)
get_filename_component(GENERATED_HDR "${GENERATED_DIRECTORY}/concrete-protocol.pb.h" ABSOLUTE)
set_source_files_properties(${GENERATED_CPP} PROPERTIES GENERATED TRUE)
set_source_files_properties(${GENERATED_HDR} PROPERTIES GENERATED TRUE)

add_custom_command(
  OUTPUT ${GENERATED_CPP} ${GENERATED_HDR}
  COMMAND ${PROTOBUF_PROTOC} --proto_path=${CONCRETE_PROTOCOL_FOLDER} --cpp_out=${GENERATED_DIRECTORY} ${CONCRETE_PROTOCOL_FILE} DEPENDS protoc
)

include_directories(${PROTOBUF_INCLUDE_DIR})
add_library(concrete-protocol STATIC
  ${GENERATED_CPP}
  ${GENERATED_HDR})
target_link_libraries(concrete-protocol PUBLIC protobuf)
