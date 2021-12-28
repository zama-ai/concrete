add_mlir_library(RTDialectAnalysis
  BufferizeDataflowTaskOps.cpp
  BuildDataflowTaskGraph.cpp
  LowerDataflowTasksToRT.cpp
  LowerRTToLLVMDFRCallsConversionPatterns.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/concretelang/Dialect/RT

  DEPENDS
  RTDialect
  AutoparPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  RTDialect)

target_link_libraries(RTDialectAnalysis PUBLIC MLIRIR)
