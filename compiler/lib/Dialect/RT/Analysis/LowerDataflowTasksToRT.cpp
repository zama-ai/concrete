// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Dialect/Concrete/IR/ConcreteDialect.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteOps.h>
#include <concretelang/Dialect/Concrete/IR/ConcreteTypes.h>
#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>
#include <concretelang/Support/math.h>
#include <mlir/IR/BuiltinOps.h>

#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
#include <llvm/IR/Instructions.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/VectorPattern.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/RegionUtils.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/RT/Analysis/Autopar.h.inc>

namespace mlir {
namespace concretelang {

namespace {

static func::FuncOp outlineWorkFunction(RT::DataflowTaskOp DFTOp,
                                        StringRef workFunctionName) {
  Location loc = DFTOp.getLoc();
  OpBuilder builder(DFTOp.getContext());
  Region &DFTOpBody = DFTOp.body();
  OpBuilder::InsertionGuard guard(builder);

  // Instead of outlining with the same operands/results, we pass all
  // results as operands as well.  For now we preserve the results'
  // types, which will be changed to use an indirection when lowering.
  SmallVector<Type, 4> operandTypes;
  operandTypes.reserve(DFTOp.getNumOperands() + DFTOp.getNumResults());
  for (Value operand : DFTOp.getOperands())
    operandTypes.push_back(RT::PointerType::get(operand.getType()));
  for (Value res : DFTOp.getResults())
    operandTypes.push_back(RT::PointerType::get(res.getType()));

  FunctionType type = FunctionType::get(DFTOp.getContext(), operandTypes, {});
  auto outlinedFunc = builder.create<func::FuncOp>(loc, workFunctionName, type);
  outlinedFunc->setAttr("_dfr_work_function_attribute", builder.getUnitAttr());
  Region &outlinedFuncBody = outlinedFunc.getBody();
  Block *outlinedEntryBlock = new Block;
  SmallVector<Location> locations(type.getInputs().size(), loc);
  outlinedEntryBlock->addArguments(type.getInputs(), locations);
  outlinedFuncBody.push_back(outlinedEntryBlock);

  BlockAndValueMapping map;
  Block &entryBlock = outlinedFuncBody.front();
  builder.setInsertionPointToStart(&entryBlock);
  for (auto operand : llvm::enumerate(DFTOp.getOperands())) {
    // Add deref of arguments and remap to operands in the body
    auto derefdop =
        builder.create<RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
            DFTOp.getLoc(), operand.value().getType(),
            entryBlock.getArgument(operand.index()));
    map.map(operand.value(), derefdop->getResult(0));
  }
  DFTOpBody.cloneInto(&outlinedFuncBody, map);

  Block &DFTOpEntry = DFTOpBody.front();
  Block *clonedDFTOpEntry = map.lookup(&DFTOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  builder.create<cf::BranchOp>(loc, clonedDFTOpEntry);

  // TODO: we use a WorkFunctionReturnOp to tie return to the
  // corresponding argument.  This can be lowered to a copy/deref for
  // shared memory and pointers, but needs to be handled for
  // distributed memory.
  outlinedFunc.walk([&](RT::DataflowYieldOp op) {
    OpBuilder replacer(op);
    int output_offset = DFTOp.getNumOperands();
    for (auto ret : llvm::enumerate(op.getOperands()))
      replacer.create<RT::WorkFunctionReturnOp>(
          op.getLoc(), ret.value(),
          outlinedFunc.getArgument(ret.index() + output_offset));
    replacer.create<func::ReturnOp>(op.getLoc());
    op.erase();
  });
  return outlinedFunc;
}

static void replaceAllUsesInDFTsInRegionWith(Value orig, Value replacement,
                                             Region &region) {
  for (auto &use : llvm::make_early_inc_range(orig.getUses())) {
    if (isa<RT::DataflowTaskOp>(use.getOwner()) &&
        region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(replacement);
  }
}
static void replaceAllUsesNotInDFTsInRegionWith(Value orig, Value replacement,
                                                Region &region) {
  for (auto &use : llvm::make_early_inc_range(orig.getUses())) {
    if (!isa<RT::DataflowTaskOp>(use.getOwner()) &&
        use.getOwner()->getParentOfType<RT::DataflowTaskOp>() == nullptr &&
        region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(replacement);
  }
}

// TODO: Fix type sizes. For now we're using some default values.
static mlir::Value getSizeInBytes(Value val, Location loc, OpBuilder builder) {
  DataLayout dataLayout = DataLayout::closest(val.getDefiningOp());
  Type type = (val.getType().isa<RT::FutureType>())
                  ? val.getType().dyn_cast<RT::FutureType>().getElementType()
                  : val.getType();

  // In the case of memref, we need to determine how much space
  // (conservatively) we need to store the memref itself. Overshooting
  // by a few bytes should not be an issue, so the main thing is to
  // properly account for the rank.
  if (type.isa<mlir::MemRefType>()) {
    // Space for the allocated and aligned pointers, and offset
    Value ptrs_offset =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(24));
    // For the sizes and shapes arrays, we need 2*8 = 16 times the rank in bytes
    Value multiplier =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(16));
    unsigned _rank = type.dyn_cast<mlir::MemRefType>().getRank();
    Value rank = builder.create<arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(_rank));
    Value sizes_shapes = builder.create<LLVM::MulOp>(loc, rank, multiplier);
    Value result = builder.create<LLVM::AddOp>(loc, ptrs_offset, sizes_shapes);
    return result;
  }

  // Unranked memrefs should be lowered to just pointer + size, so we need 16
  // bytes.
  if (type.isa<mlir::UnrankedMemRefType>())
    return builder.create<arith::ConstantOp>(loc,
                                             builder.getI64IntegerAttr(16));

  // FHE types are converted to pointers, so we take their size as 8
  // bytes until we can get the actual size of the actual types.
  if (type.isa<mlir::concretelang::Concrete::ContextType>() ||
      type.isa<mlir::concretelang::Concrete::LweCiphertextType>() ||
      type.isa<mlir::concretelang::Concrete::GlweCiphertextType>())
    return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(8));

  // For all other types, get type size.
  return builder.create<arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(dataLayout.getTypeSize(type)));
}

static void lowerDataflowTaskOp(RT::DataflowTaskOp DFTOp,
                                func::FuncOp workFunction) {
  DataLayout dataLayout = DataLayout::closest(DFTOp);
  Region &opBody = DFTOp->getParentOfType<func::FuncOp>().getBody();
  BlockAndValueMapping map;
  OpBuilder builder(DFTOp);

  // First identify DFT operands that are not futures and are not
  // defined by another DFT. These need to be made into futures and
  // propagated to all other DFTs. We can allow PRE to eliminate the
  // previous definitions if there are no non-future type uses.
  builder.setInsertionPoint(DFTOp);
  for (Value val : DFTOp.getOperands()) {
    if (!val.getType().isa<RT::FutureType>()) {
      Type futType = RT::FutureType::get(val.getType());
      auto mrf =
          builder.create<RT::MakeReadyFutureOp>(DFTOp.getLoc(), futType, val);
      map.map(mrf->getResult(0), val);
      replaceAllUsesInDFTsInRegionWith(val, mrf->getResult(0), opBody);
    }
  }

  // Second generate a CreateAsyncTaskOp that will replace the
  // DataflowTaskOp.  This also includes the necessary handling of
  // operands and results (conversion to/from futures and propagation).
  SmallVector<Value, 4> catOperands;
  int size = 3 + DFTOp.getNumResults() * 2 + DFTOp.getNumOperands() * 2;
  catOperands.reserve(size);
  auto fnptr = builder.create<mlir::func::ConstantOp>(
      DFTOp.getLoc(), workFunction.getFunctionType(),
      SymbolRefAttr::get(builder.getContext(), workFunction.getName()));
  auto numIns = builder.create<arith::ConstantOp>(
      DFTOp.getLoc(), builder.getI64IntegerAttr(DFTOp.getNumOperands()));
  auto numOuts = builder.create<arith::ConstantOp>(
      DFTOp.getLoc(), builder.getI64IntegerAttr(DFTOp.getNumResults()));
  catOperands.push_back(fnptr.getResult());
  catOperands.push_back(numIns.getResult());
  catOperands.push_back(numOuts.getResult());
  for (auto operand : DFTOp.getOperands()) {
    catOperands.push_back(operand);
    catOperands.push_back(getSizeInBytes(operand, DFTOp.getLoc(), builder));
  }

  // We need to adjust the results for the CreateAsyncTaskOp which
  // are the work function's returns through pointers passed as
  // parameters.  As this is not supported within MLIR - and mostly
  // unsupported even in the LLVMIR Dialect - this needs to use two
  // placeholders for each output, before and after the
  // CreateAsyncTaskOp.
  for (auto result : DFTOp.getResults()) {
    Type futType = RT::PointerType::get(RT::FutureType::get(result.getType()));
    auto brpp = builder.create<RT::BuildReturnPtrPlaceholderOp>(DFTOp.getLoc(),
                                                                futType);
    map.map(result, brpp->getResult(0));
    catOperands.push_back(brpp->getResult(0));
    catOperands.push_back(getSizeInBytes(result, DFTOp.getLoc(), builder));
  }
  builder.create<RT::CreateAsyncTaskOp>(
      DFTOp.getLoc(),
      SymbolRefAttr::get(builder.getContext(), workFunction.getName()),
      catOperands);

  // Third identify results of this DFT that are not used *only* in
  // other DFTs as those will need to be waited on explicitly.
  // We also create the DerefReturnPtrPlaceholderOp after the
  // CreateAsyncTaskOp.  These also need propagating.
  for (auto result : DFTOp.getResults()) {
    Type futType = RT::FutureType::get(result.getType());
    Value futptr = map.lookupOrNull(result);
    assert(futptr);
    auto drpp = builder.create<RT::DerefReturnPtrPlaceholderOp>(
        DFTOp.getLoc(), futType, futptr);
    replaceAllUsesInDFTsInRegionWith(result, drpp->getResult(0), opBody);

    for (auto &use : llvm::make_early_inc_range(result.getUses())) {
      if (!isa<RT::DataflowTaskOp>(use.getOwner()) &&
          use.getOwner()->getParentOfType<RT::DataflowTaskOp>() == nullptr) {
        // Wait for this future
        // TODO: the wait function should ideally
        // be issued as late as possible, but need to identify which
        // use comes first.
        auto af = builder.create<RT::AwaitFutureOp>(
            DFTOp.getLoc(), result.getType(), drpp.getResult());
        replaceAllUsesNotInDFTsInRegionWith(result, af->getResult(0), opBody);
        // We only need to to this once, propagation will hit all
        // other uses
        break;
      }
    }
    // All leftover uses (i.e. those within DFTs should use the future)
    replaceAllUsesInRegionWith(result, futptr, opBody);
  }

  // Finally erase the DFT.
  DFTOp.erase();
}

/// For documentation see Autopar.td
struct LowerDataflowTasksPass
    : public LowerDataflowTasksBase<LowerDataflowTasksPass> {

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](mlir::func::FuncOp func) {
      static int wfn_id = 0;

      // TODO: For now do not attempt to use nested parallelism.
      if (func->getAttr("_dfr_work_function_attribute"))
        return;

      SymbolTable symbolTable = mlir::SymbolTable::getNearestSymbolTable(func);
      std::vector<std::pair<RT::DataflowTaskOp, func::FuncOp>> outliningMap;

      func.walk([&](RT::DataflowTaskOp op) {
        auto workFunctionName =
            Twine("_dfr_DFT_work_function__") +
            Twine(op->getParentOfType<func::FuncOp>().getName()) +
            Twine(wfn_id++);
        func::FuncOp outlinedFunc =
            outlineWorkFunction(op, workFunctionName.str());
        outliningMap.push_back(
            std::pair<RT::DataflowTaskOp, func::FuncOp>(op, outlinedFunc));
        symbolTable.insert(outlinedFunc);
        return WalkResult::advance();
      });

      // Lower the DF task ops to RT dialect ops.
      for (auto mapping : outliningMap)
        lowerDataflowTaskOp(mapping.first, mapping.second);

      // Issue _dfr_start/stop calls for this function
      if (!outliningMap.empty()) {
        OpBuilder builder(func.getBody());
        builder.setInsertionPointToStart(&func.getBody().front());
        auto dfrStartFunOp = mlir::LLVM::lookupOrCreateFn(
            func->getParentOfType<ModuleOp>(), "_dfr_start", {},
            LLVM::LLVMVoidType::get(func->getContext()));
        builder.create<LLVM::CallOp>(func.getLoc(), dfrStartFunOp,
                                     mlir::ValueRange(),
                                     ArrayRef<NamedAttribute>());

        builder.setInsertionPoint(func.getBody().back().getTerminator());
        auto dfrStopFunOp = mlir::LLVM::lookupOrCreateFn(
            func->getParentOfType<ModuleOp>(), "_dfr_stop", {},
            LLVM::LLVMVoidType::get(func->getContext()));
        builder.create<LLVM::CallOp>(func.getLoc(), dfrStopFunOp,
                                     mlir::ValueRange(),
                                     ArrayRef<NamedAttribute>());
      }
    });

    // Delay memref deallocations when memrefs are made into futures
    module.walk([&](Operation *op) {
      if (isa<RT::MakeReadyFutureOp>(*op) &&
          op->getOperand(0).getType().isa<mlir::MemRefType>()) {
        for (auto &use :
             llvm::make_early_inc_range(op->getOperand(0).getUses())) {
          if (isa<mlir::memref::DeallocOp>(use.getOwner())) {
            OpBuilder builder(use.getOwner()
                                  ->getParentOfType<mlir::func::FuncOp>()
                                  .getBody()
                                  .back()
                                  .getTerminator());
            builder.clone(*use.getOwner());
            use.getOwner()->erase();
          }
        }
      }
      return WalkResult::advance();
    });
  }
  LowerDataflowTasksPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLowerDataflowTasksPass(bool debug) {
  return std::make_unique<LowerDataflowTasksPass>(debug);
}

} // end namespace concretelang
} // end namespace mlir
