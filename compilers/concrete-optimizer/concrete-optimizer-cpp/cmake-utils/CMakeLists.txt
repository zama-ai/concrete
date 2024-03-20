# In your project CMakeLists.txt:
# 1. define the path containing concrete-optimizer repository, built.
# CONCRETE_OPTIMIZER_DIR
# 2. add lines:
# add_subdirectory(${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/cmake-utils)
# include_directories(${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/src/cpp)

add_library(concrete_optimizer STATIC ${CONCRETE_OPTIMIZER_DIR}/concrete-optimizer-cpp/src/cpp/concrete-optimizer.cpp)

target_link_libraries(concrete_optimizer PRIVATE
  pthread m dl
  "${CONCRETE_OPTIMIZER_DIR}/target/libconcrete_optimizer_cpp.a"
)

install(TARGETS concrete_optimizer EXPORT concrete_optimizer)
install(EXPORT concrete_optimizer DESTINATION "./")
