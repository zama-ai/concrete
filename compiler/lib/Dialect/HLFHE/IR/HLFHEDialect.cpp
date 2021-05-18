#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"

using namespace mlir::zamalang::HLFHE;

void HLFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.cpp.inc"
      >();
}
