cmake_minimum_required(VERSION 3.17)

project(concrete-protocol CXX)
include(ExternalProject)

set(CAPNP_VERSION 1.0.1)
set(CAPNP_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/capnp_bin_dir)
set(CAPNP_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/capnp_src_dir)
set(CAPNP_LIB_DIR ${CAPNP_BIN_DIR}/c++/src)
set(CAPNP_INCLUDE_DIR ${CAPNP_SRC_DIR}/c++/src)
file(MAKE_DIRECTORY "${CAPNP_BIN_DIR}")
set(CAPNP_CMD ${CAPNP_BIN_DIR}/c++/src/capnp/capnp)
set(CAPNP_LIBRARY libcapnp.a)
set(CAPNP_JSON_LIBRARY libcapnp-json.a)
set(KJ_LIBRARY libkj.a)

ExternalProject_Add(
  capnp_repo
  GIT_REPOSITORY https://github.com/capnproto/capnproto.git
  GIT_TAG release-${CAPNP_VERSION}
  GIT_SUBMODULES_RECURSE ON
  GIT_PROGRESS   TRUE
  UPDATE_DISCONNECTED TRUE
  BUILD_ALWAYS 1
  INSTALL_COMMAND cp ${CAPNP_LIB_DIR}/capnp/${CAPNP_LIBRARY} ${CAPNP_LIB_DIR}/capnp/${CAPNP_JSON_LIBRARY} ${CAPNP_LIB_DIR}/kj/${KJ_LIBRARY} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  BINARY_DIR ${CAPNP_BIN_DIR}
  SOURCE_DIR ${CAPNP_SRC_DIR}
  CMAKE_ARGS -Dcapnp_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  BUILD_BYPRODUCTS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CAPNP_LIBRARY} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CAPNP_JSON_LIBRARY} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${KJ_LIBRARY} ${CAPNP_CMD}
)

add_library(kj STATIC IMPORTED GLOBAL)
set_property(TARGET kj PROPERTY IMPORTED_LOCATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${KJ_LIBRARY})
add_dependencies(kj capnp_repo)

add_library(capnp STATIC IMPORTED GLOBAL)
set_property(TARGET capnp PROPERTY IMPORTED_LOCATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CAPNP_LIBRARY})

add_dependencies(capnp capnp_repo)

add_library(capnp-json STATIC IMPORTED GLOBAL)
set_property(TARGET capnp-json PROPERTY IMPORTED_LOCATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CAPNP_JSON_LIBRARY})
add_dependencies(capnp-json capnp_repo)

add_executable(capnpc IMPORTED)
set_property(TARGET capnpc PROPERTY IMPORTED_LOCATION ${CAPNP_CMD})
add_dependencies(capnpc capnp_repo)

get_filename_component(CONCRETE_PROTOCOL_CAPNP_FILE "src/concrete-protocol.capnp" ABSOLUTE)
get_filename_component(CONCRETE_PROTOCOL_FOLDER "${CONCRETE_PROTOCOL_CAPNP_FILE}" DIRECTORY)
set(GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
file(MAKE_DIRECTORY "${GENERATED_DIRECTORY}")
get_filename_component(CAPNP_GENERATED_CPP "${GENERATED_DIRECTORY}/concrete-protocol.capnp.c++" ABSOLUTE)
get_filename_component(CAPNP_GENERATED_HDR "${GENERATED_DIRECTORY}/concrete-protocol.capnp.h" ABSOLUTE)
set_source_files_properties(${CAPNP_GENERATED_CPP} PROPERTIES GENERATED TRUE)
set_source_files_properties(${CAPNP_GENERATED_HDR} PROPERTIES GENERATED TRUE)

add_custom_command(
  OUTPUT ${CAPNP_GENERATED_CPP} ${CAPNP_GENERATED_HDR}
  COMMAND ${CAPNP_CMD} compile --src-prefix=${CONCRETE_PROTOCOL_FOLDER} --import-path=${CAPNP_INCLUDE_DIR} --output=${CAPNP_CMD}c-c++:${GENERATED_DIRECTORY} ${CONCRETE_PROTOCOL_CAPNP_FILE} DEPENDS capnpc
)

include_directories(${CAPNP_INCLUDE_DIR})
add_library(concrete-protocol STATIC
  ${CAPNP_GENERATED_CPP}
  ${CAPNP_GENERATED_HDR})
target_link_libraries(concrete-protocol PRIVATE capnp capnp-json kj)
