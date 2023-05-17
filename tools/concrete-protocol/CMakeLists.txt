
cmake_minimum_required(VERSION 3.17)

project(concrete-protocol CXX)



get_filename_component(PROTO_FILE "src/concrete-protocol.proto" ABSOLUTE)
get_filename_component(PROTO_FOLDER "${PROTO_FILE}" DIRECTORY)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(GEN_CPP GEN_HDR ${PROTO_FILE})

set_source_files_properties(${GEN_CPP} PROPERTIES GENERATED TRUE)
set_source_files_properties(${GEN_HDR} PROPERTIES GENERATED TRUE)

add_library(concrete-protocol STATIC
  ${GEN_CPP}
  ${GEN_HDR})
target_compile_options(concrete-protocol PRIVATE "-Wno-c++98-compat-extra-semi")
