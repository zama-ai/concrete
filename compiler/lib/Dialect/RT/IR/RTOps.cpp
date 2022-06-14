// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"

#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/RT/IR/RTTypes.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/RT/IR/RTOps.cpp.inc"

using namespace mlir::concretelang::RT;

void DataflowTaskOp::build(
    ::mlir::OpBuilder &builder, ::mlir::OperationState &result,
    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  result.addOperands(operands);
  result.addAttributes(attributes);
  Region *reg = result.addRegion();
  Block *body = new Block();
  reg->push_back(body);
  result.addTypes(resultTypes);
}

void DataflowTaskOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {}
