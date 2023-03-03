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
#include "llvm/ADT/STLExtras.h"

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

llvm::Optional<mlir::Operation *>
DataflowTaskOp::buildDealloc(OpBuilder &builder, Value alloc) {
  return builder.create<DeallocateFutureOp>(alloc.getLoc(), alloc)
      .getOperation();
}
llvm::Optional<mlir::Value> DataflowTaskOp::buildClone(OpBuilder &builder,
                                                       Value alloc) {
  return builder.create<CloneFutureOp>(alloc.getLoc(), alloc).getResult();
}
void DataflowTaskOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto input : inputs())
    effects.emplace_back(MemoryEffects::Read::get(), input,
                         SideEffects::DefaultResource::get());
  for (auto output : outputs())
    effects.emplace_back(MemoryEffects::Write::get(), output,
                         SideEffects::DefaultResource::get());
  for (auto output : outputs())
    effects.emplace_back(MemoryEffects::Allocate::get(), output,
                         SideEffects::DefaultResource::get());
}

llvm::Optional<mlir::Operation *>
CloneFutureOp::buildDealloc(OpBuilder &builder, Value alloc) {
  return builder.create<DeallocateFutureOp>(alloc.getLoc(), alloc)
      .getOperation();
}
llvm::Optional<mlir::Value> CloneFutureOp::buildClone(OpBuilder &builder,
                                                      Value alloc) {
  return builder.create<CloneFutureOp>(alloc.getLoc(), alloc).getResult();
}
void CloneFutureOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), input(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), output(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), output(),
                       SideEffects::DefaultResource::get());
}

llvm::Optional<mlir::Operation *>
MakeReadyFutureOp::buildDealloc(OpBuilder &builder, Value alloc) {
  return builder.create<DeallocateFutureOp>(alloc.getLoc(), alloc)
      .getOperation();
}
llvm::Optional<mlir::Value> MakeReadyFutureOp::buildClone(OpBuilder &builder,
                                                          Value alloc) {
  return builder.create<CloneFutureOp>(alloc.getLoc(), alloc).getResult();
}
void MakeReadyFutureOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), input(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), output(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), output(),
                       SideEffects::DefaultResource::get());
}
