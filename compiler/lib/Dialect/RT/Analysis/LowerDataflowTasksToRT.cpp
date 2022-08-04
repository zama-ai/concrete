// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Conversion/Tools.h>
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
#include <concretelang/Runtime/DFRuntime.hpp>
#include <concretelang/Support/math.h>
#include <mlir/IR/BuiltinOps.h>

#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
#include <llvm/IR/Instructions.h>
#include <mlir/Analysis/DataFlowAnalysis.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/VectorPattern.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
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
    if ((isa<RT::DataflowTaskOp>(use.getOwner()) ||
         isa<RT::DeallocateFutureOp>(use.getOwner())) &&
        region.isAncestor(use.getOwner()->getParentRegion()))
      use.set(replacement);
  }
}

// TODO: Fix type sizes. For now we're using some default values.
static std::pair<mlir::Value, mlir::Value>
getTaskArgumentSizeAndType(Value val, Location loc, OpBuilder builder) {
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
    Value typeSize =
        builder.create<LLVM::AddOp>(loc, ptrs_offset, sizes_shapes);

    Type elementType = type.dyn_cast<mlir::MemRefType>().getElementType();
    // Assume here that the base type is a simple scalar-type or at
    // least its size can be determined.
    // size_t elementAttr = dataLayout.getTypeSize(elementType);
    // Make room for a byte to store the type of this argument/output
    // elementAttr <<= 8;
    // elementAttr |= _DFR_TASK_ARG_MEMREF;
    uint64_t elementAttr = 0;
    size_t element_size = dataLayout.getTypeSize(elementType);
    elementAttr =
        dfr::_dfr_set_arg_type(elementAttr, dfr::_DFR_TASK_ARG_MEMREF);
    elementAttr = dfr::_dfr_set_memref_element_size(elementAttr, element_size);
    Value arg_type = builder.create<arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(elementAttr));
    return std::pair<mlir::Value, mlir::Value>(typeSize, arg_type);
  }

  // Unranked memrefs should be lowered to just pointer + size, so we need 16
  // bytes.
  assert(!type.isa<mlir::UnrankedMemRefType>() &&
         "UnrankedMemRefType not currently supported");

  Value arg_type = builder.create<arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(dfr::_DFR_TASK_ARG_BASE));

  // FHE types are converted to pointers, so we take their size as 8
  // bytes until we can get the actual size of the actual types.
  if (type.isa<mlir::concretelang::Concrete::LweCiphertextType>() ||
      type.isa<mlir::concretelang::Concrete::GlweCiphertextType>() ||
      type.isa<mlir::concretelang::Concrete::PlaintextType>()) {
    Value result =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(8));
    return std::pair<mlir::Value, mlir::Value>(result, arg_type);
  } else if (type.isa<mlir::concretelang::Concrete::ContextType>()) {
    Value arg_type = builder.create<arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(dfr::_DFR_TASK_ARG_CONTEXT));
    Value result =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(8));
    return std::pair<mlir::Value, mlir::Value>(result, arg_type);
  }

  // For all other types, get type size.
  Value result = builder.create<arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(dataLayout.getTypeSize(type)));
  return std::pair<mlir::Value, mlir::Value>(result, arg_type);
}

static void getAliasedUses(Value val, DenseSet<OpOperand *> &aliasedUses) {
  for (auto &use : val.getUses()) {
    aliasedUses.insert(&use);
    if (isa<memref::CastOp, memref::ViewOp, memref::SubViewOp>(use.getOwner()))
      getAliasedUses(use.getOwner()->getResult(0), aliasedUses);
  }
}

static void lowerDataflowTaskOp(RT::DataflowTaskOp DFTOp,
                                func::FuncOp workFunction) {
  DataLayout dataLayout = DataLayout::closest(DFTOp);
  Region &opBody = DFTOp->getParentOfType<func::FuncOp>().getBody();
  OpBuilder builder(DFTOp);

  // First identify DFT operands that are not futures and are not
  // defined by another DFT. These need to be made into futures and
  // propagated to all other DFTs. We can allow PRE to eliminate the
  // previous definitions if there are no non-future type uses.
  for (Value val : DFTOp.getOperands()) {
    if (!val.getType().isa<RT::FutureType>()) {
      OpBuilder::InsertionGuard guard(builder);
      Type futType = RT::FutureType::get(val.getType());
      Value memrefCloned, newval = val;

      // Find out if this value is needed in any other task
      SmallVector<Operation *, 2> taskOps;
      for (auto &use : val.getUses())
        if (isa<RT::DataflowTaskOp>(use.getOwner()))
          taskOps.push_back(use.getOwner());
      Operation *first = DFTOp;
      for (auto op : taskOps)
        if (first->getBlock() == op->getBlock() && op->isBeforeInBlock(first))
          first = op;
      builder.setInsertionPoint(first);

      // If we are building a future on a MemRef, then we need to clone
      // the memref in order to allow the deallocation pass which does
      // not synchronize with task execution.
      if (val.getType().isa<mlir::MemRefType>()) {
        // Get the type of memref that we will clone.  In case this is
        // a subview, we discard the mapping so we copy to a contiguous
        // layout which pre-serializes this.
        MemRefType mrType = val.getType().dyn_cast<mlir::MemRefType>();
        if (!mrType.getLayout().isIdentity()) {
          unsigned rank = mrType.getRank();
          mrType = MemRefType::Builder(mrType)
                       .setShape(mrType.getShape())
                       .setLayout(AffineMapAttr::get(
                           builder.getMultiDimIdentityMap(rank)));
        }
        newval = builder.create<mlir::memref::AllocOp>(val.getLoc(), mrType)
                     .getResult();
        builder.create<mlir::memref::CopyOp>(val.getLoc(), val, newval);
        memrefCloned = builder.create<arith::ConstantOp>(
            val.getLoc(), builder.getI64IntegerAttr(1));
      } else {
        memrefCloned = builder.create<arith::ConstantOp>(
            val.getLoc(), builder.getI64IntegerAttr(0));
      }

      auto mrf = builder.create<RT::MakeReadyFutureOp>(val.getLoc(), futType,
                                                       newval, memrefCloned);
      replaceAllUsesInDFTsInRegionWith(val, mrf, opBody);
    }
  }

  // Second generate a CreateAsyncTaskOp that will replace the
  // DataflowTaskOp.  This also includes the necessary handling of
  // operands and results (conversion to/from futures and propagation).
  SmallVector<Value, 4> catOperands;
  int size = 3 + DFTOp.getNumResults() * 3 + DFTOp.getNumOperands() * 3;
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
    auto op_size = getTaskArgumentSizeAndType(operand, DFTOp.getLoc(), builder);
    catOperands.push_back(operand);
    catOperands.push_back(op_size.first);
    catOperands.push_back(op_size.second);
  }

  // We need to adjust the results for the CreateAsyncTaskOp which
  // are the work function's returns through pointers passed as
  // parameters.  As this is not supported within MLIR - and mostly
  // unsupported even in the LLVMIR Dialect - this needs to use two
  // placeholders for each output, before and after the
  // CreateAsyncTaskOp.
  BlockAndValueMapping map;
  for (auto result : DFTOp.getResults()) {
    Type futType = RT::PointerType::get(RT::FutureType::get(result.getType()));
    auto brpp = builder.create<RT::BuildReturnPtrPlaceholderOp>(DFTOp.getLoc(),
                                                                futType);
    auto op_size = getTaskArgumentSizeAndType(result, DFTOp.getLoc(), builder);
    map.map(result, brpp->getResult(0));
    catOperands.push_back(brpp->getResult(0));
    catOperands.push_back(op_size.first);
    catOperands.push_back(op_size.second);
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
          !isa<RT::DeallocateFutureOp>(use.getOwner()) &&
          use.getOwner()->getParentOfType<RT::DataflowTaskOp>() == nullptr) {
        // Wait for this future before its uses
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(use.getOwner());
        auto af = builder.create<RT::AwaitFutureOp>(
            DFTOp.getLoc(), result.getType(), drpp.getResult());
        assert(opBody.isAncestor(use.getOwner()->getParentRegion()));
        use.set(af->getResult(0));
      }
    }
    // All leftover uses (i.e. those within DFTs should use the future)
    replaceAllUsesInRegionWith(result, futptr, opBody);
  }

  // Finally erase the DFT.
  DFTOp.erase();
}

static void registerWorkFunction(mlir::func::FuncOp parentFunc,
                                 mlir::func::FuncOp workFunction) {
  OpBuilder builder(parentFunc.getBody());
  builder.setInsertionPointToStart(&parentFunc.getBody().front());

  auto fnptr = builder.create<mlir::func::ConstantOp>(
      parentFunc.getLoc(), workFunction.getFunctionType(),
      SymbolRefAttr::get(builder.getContext(), workFunction.getName()));

  builder.create<RT::RegisterTaskWorkFunctionOp>(parentFunc.getLoc(),
                                                 fnptr.getResult());
}

static func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

/// For documentation see Autopar.td
struct LowerDataflowTasksPass
    : public LowerDataflowTasksBase<LowerDataflowTasksPass> {

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> workFunctions;
    SmallVector<func::FuncOp, 1> entryPoints;

    module.walk([&](mlir::func::FuncOp func) {
      static int wfn_id = 0;

      // TODO: For now do not attempt to use nested parallelism.
      if (func->getAttr("_dfr_work_function_attribute"))
        return;

      SymbolTable symbolTable = mlir::SymbolTable::getNearestSymbolTable(func);
      SmallVector<std::pair<RT::DataflowTaskOp, func::FuncOp>, 4> outliningMap;

      func.walk([&](RT::DataflowTaskOp op) {
        auto workFunctionName =
            Twine("_dfr_DFT_work_function__") +
            Twine(op->getParentOfType<func::FuncOp>().getName()) +
            Twine(wfn_id++);
        func::FuncOp outlinedFunc =
            outlineWorkFunction(op, workFunctionName.str());
        outliningMap.push_back(
            std::pair<RT::DataflowTaskOp, func::FuncOp>(op, outlinedFunc));
        workFunctions.push_back(outlinedFunc);
        symbolTable.insert(outlinedFunc);
        return WalkResult::advance();
      });

      // Lower the DF task ops to RT dialect ops.
      for (auto mapping : outliningMap)
        lowerDataflowTaskOp(mapping.first, mapping.second);

      // Main is always an entry-point - otherwise check if this
      // function is called within the module.  TODO: we assume no
      // recursion.
      if (func.getName() == "main")
        entryPoints.push_back(func);
      else {
        bool found = false;
        module.walk([&](mlir::func::CallOp op) {
          if (getCalledFunction(op) == func)
            found = true;
        });
        if (!found)
          entryPoints.push_back(func);
      }
    });

    for (auto entryPoint : entryPoints) {
      // Check if this entry point uses a context - do this before we
      // remove arguments in remote nodes
      int ctxIndex = -1;
      for (auto arg : llvm::enumerate(entryPoint.getArguments()))
        if (arg.value()
                .getType()
                .isa<mlir::concretelang::Concrete::ContextType>()) {
          ctxIndex = arg.index();
          break;
        }

      // If this is a JIT invocation and we're not on the root node,
      // we do not need to do any computation, only register all work
      // functions with the runtime system
      if (!workFunctions.empty()) {
        if (!dfr::_dfr_is_root_node()) {
          entryPoint.eraseBody();
          Block *b = new Block;
          FunctionType funTy = entryPoint.getFunctionType();
          SmallVector<Location> locations(funTy.getInputs().size(),
                                          entryPoint.getLoc());
          b->addArguments(funTy.getInputs(), locations);
          entryPoint.getBody().push_front(b);
          for (int i = funTy.getNumInputs() - 1; i >= 0; --i)
            entryPoint.eraseArgument(i);
          for (int i = funTy.getNumResults() - 1; i >= 0; --i)
            entryPoint.eraseResult(i);
          OpBuilder builder(entryPoint.getBody());
          builder.setInsertionPointToEnd(&entryPoint.getBody().front());
          builder.create<mlir::func::ReturnOp>(entryPoint.getLoc());
        }
      }

      // Generate code to register all work-functions with the
      // runtime.
      for (auto wf : workFunctions)
        registerWorkFunction(entryPoint, wf);

      // Issue _dfr_start/stop calls for this function
      OpBuilder builder(entryPoint.getBody());
      builder.setInsertionPointToStart(&entryPoint.getBody().front());
      int useDFR = (workFunctions.empty()) ? 0 : 1;
      Value useDFRVal = builder.create<arith::ConstantOp>(
          entryPoint.getLoc(), builder.getI64IntegerAttr(useDFR));

      if (ctxIndex >= 0) {
        auto startFunTy =
            (dfr::_dfr_is_root_node())
                ? mlir::FunctionType::get(
                      entryPoint->getContext(),
                      {useDFRVal.getType(),
                       entryPoint.getArgument(ctxIndex).getType()},
                      {})
                : mlir::FunctionType::get(entryPoint->getContext(),
                                          {useDFRVal.getType()}, {});
        (void)insertForwardDeclaration(entryPoint, builder, "_dfr_start_c",
                                       startFunTy);
        (dfr::_dfr_is_root_node())
            ? builder.create<mlir::func::CallOp>(
                  entryPoint.getLoc(), "_dfr_start_c", mlir::TypeRange(),
                  mlir::ValueRange(
                      {useDFRVal, entryPoint.getArgument(ctxIndex)}))
            : builder.create<mlir::func::CallOp>(entryPoint.getLoc(),
                                                 "_dfr_start_c",
                                                 mlir::TypeRange(), useDFRVal);
      } else {
        auto startFunTy = mlir::FunctionType::get(entryPoint->getContext(),
                                                  {useDFRVal.getType()}, {});
        (void)insertForwardDeclaration(entryPoint, builder, "_dfr_start",
                                       startFunTy);
        builder.create<mlir::func::CallOp>(entryPoint.getLoc(), "_dfr_start",
                                           mlir::TypeRange(), useDFRVal);
      }
      builder.setInsertionPoint(entryPoint.getBody().back().getTerminator());
      auto stopFunTy = mlir::FunctionType::get(entryPoint->getContext(),
                                               {useDFRVal.getType()}, {});
      (void)insertForwardDeclaration(entryPoint, builder, "_dfr_stop",
                                     stopFunTy);
      builder.create<mlir::func::CallOp>(entryPoint.getLoc(), "_dfr_stop",
                                         mlir::TypeRange(), useDFRVal);
    }
  }
  LowerDataflowTasksPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createLowerDataflowTasksPass(bool debug) {
  return std::make_unique<LowerDataflowTasksPass>(debug);
}

namespace {

// For documentation see Autopar.td
struct FixupBufferDeallocationPass
    : public FixupBufferDeallocationBase<FixupBufferDeallocationPass> {

  void runOnOperation() override {
    auto module = getOperation();
    std::vector<Operation *> ops;

    // All buffers allocated and either made into a future, directly
    // or as a result of being returned by a task, are managed by the
    // DFR runtime system's reference counting.
    module.walk([&](RT::WorkFunctionReturnOp retOp) {
      for (auto &use :
           llvm::make_early_inc_range(retOp.getOperands().front().getUses()))
        if (isa<mlir::memref::DeallocOp>(use.getOwner()))
          ops.push_back(use.getOwner());
    });
    module.walk([&](RT::MakeReadyFutureOp mrfOp) {
      for (auto &use :
           llvm::make_early_inc_range(mrfOp.getOperands().front().getUses()))
        if (isa<mlir::memref::DeallocOp>(use.getOwner()))
          ops.push_back(use.getOwner());
    });
    for (auto op : ops)
      op->erase();
  }
  FixupBufferDeallocationPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createFixupBufferDeallocationPass(bool debug) {
  return std::make_unique<FixupBufferDeallocationPass>(debug);
}

} // end namespace concretelang
} // end namespace mlir
