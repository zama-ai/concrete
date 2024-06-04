// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Conversion/Tools.h>
#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
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

#include <llvm/IR/Instructions.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/VectorPattern.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
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
  Region &DFTOpBody = DFTOp.getBody();
  OpBuilder::InsertionGuard guard(builder);

  SetVector<Value> operands;
  getUsedValuesDefinedAbove(DFTOpBody, operands);
  DFTOp->setOperands(operands.takeVector());

  // Instead of outlining with the same operands/results, we pass all
  // results as operands as well.  For now we preserve the results'
  // types, which will be changed to use an indirection when lowering.
  SmallVector<Type, 4> operandTypes;
  operandTypes.reserve(DFTOp.getNumOperands() + DFTOp.getNumResults());
  for (Value res : DFTOp.getResults())
    operandTypes.push_back(RT::PointerType::get(res.getType()));
  for (Value operand : DFTOp.getOperands())
    operandTypes.push_back(RT::PointerType::get(operand.getType()));

  FunctionType type = FunctionType::get(DFTOp.getContext(), operandTypes, {});
  auto outlinedFunc = builder.create<func::FuncOp>(loc, workFunctionName, type);
  outlinedFunc->setAttr("_dfr_work_function_attribute", builder.getUnitAttr());
  Region &outlinedFuncBody = outlinedFunc.getBody();
  Block *outlinedEntryBlock = new Block;
  SmallVector<Location> locations(type.getInputs().size(), loc);
  outlinedEntryBlock->addArguments(type.getInputs(), locations);
  outlinedFuncBody.push_back(outlinedEntryBlock);

  IRMapping map;
  int input_offset = DFTOp.getNumResults();
  Block &entryBlock = outlinedFuncBody.front();
  builder.setInsertionPointToStart(&entryBlock);
  for (auto operand : llvm::enumerate(DFTOp.getOperands())) {
    // Add deref of arguments and remap to operands in the body
    auto derefdop =
        builder.create<RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
            DFTOp.getLoc(), operand.value().getType(),
            entryBlock.getArgument(operand.index() + input_offset));
    map.map(operand.value(), derefdop->getResult(0));
  }
  DFTOpBody.cloneInto(&outlinedFuncBody, map);

  Block &DFTOpEntry = DFTOpBody.front();
  Block *clonedDFTOpEntry = map.lookup(&DFTOpEntry);
  builder.setInsertionPointToEnd(&entryBlock);
  builder.create<cf::BranchOp>(loc, clonedDFTOpEntry);

  // WorkFunctionReturnOp ties return to the corresponding argument.
  // This is lowered to a copy/deref for shared memory and pointers,
  // and handled in the serializer for distributed memory.
  outlinedFunc.walk([&](RT::DataflowYieldOp op) {
    OpBuilder replacer(op);
    for (auto ret : llvm::enumerate(op.getOperands()))
      replacer.create<RT::WorkFunctionReturnOp>(
          op.getLoc(), ret.value(), outlinedFunc.getArgument(ret.index()));
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

static mlir::Type stripType(mlir::Type type) {
  if (type.isa<RT::FutureType>())
    return stripType(type.dyn_cast<RT::FutureType>().getElementType());
  if (type.isa<RT::PointerType>())
    return stripType(type.dyn_cast<RT::PointerType>().getElementType());
  return type;
}

// TODO: Fix type sizes. For now we're using some default values.
static std::pair<Value, Value>
getTaskArgumentSizeAndType(Value val, Location loc, OpBuilder builder) {
  DataLayout dataLayout = DataLayout::closest(val.getDefiningOp());
  Type type = stripType(val.getType());

  // In the case of memref, we need to determine how much space
  // (conservatively) we need to store the memref itself. Overshooting
  // by a few bytes should not be an issue, so the main thing is to
  // properly account for the rank.
  if (type.isa<mlir::MemRefType>()) {
    // Space for the allocated and aligned pointers, and offset plus
    // rank * sizes and strides
    size_t element_size;
    unsigned rank = type.dyn_cast<mlir::MemRefType>().getRank();
    Type elementType = type.dyn_cast<mlir::MemRefType>().getElementType();

    element_size = dataLayout.getTypeSize(elementType);

    size_t size = 24 + 16 * rank;
    Value typeSize =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(size));

    // Assume here that the base type is a simple scalar-type or at
    // least its size can be determined.
    // size_t elementAttr = dataLayout.getTypeSize(elementType);
    // Make room for a byte to store the type of this argument/output
    // elementAttr <<= 8;
    // elementAttr |= _DFR_TASK_ARG_MEMREF;
    uint64_t elementAttr = 0;
    elementAttr =
        dfr::_dfr_set_arg_type(elementAttr, dfr::_DFR_TASK_ARG_MEMREF);
    elementAttr = dfr::_dfr_set_memref_element_size(elementAttr, element_size);
    Value arg_type = builder.create<arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(elementAttr));
    return std::pair<mlir::Value, mlir::Value>(typeSize, arg_type);
  }

  if (type.isa<mlir::concretelang::Concrete::ContextType>()) {
    Value arg_type = builder.create<arith::ConstantOp>(
        loc, builder.getI64IntegerAttr(dfr::_DFR_TASK_ARG_CONTEXT));
    Value typeSize =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(8));
    return std::pair<mlir::Value, mlir::Value>(typeSize, arg_type);
  }

  Value arg_type = builder.create<arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(dfr::_DFR_TASK_ARG_BASE));
  Value typeSize = builder.create<arith::ConstantOp>(
      loc, builder.getI64IntegerAttr(dataLayout.getTypeSize(type)));
  return std::pair<mlir::Value, mlir::Value>(typeSize, arg_type);
}

static void lowerDataflowTaskOp(RT::DataflowTaskOp DFTOp,
                                func::FuncOp workFunction) {
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
      auto mrf = builder.create<RT::MakeReadyFutureOp>(
          val.getLoc(), futType, val,
          builder.create<arith::ConstantOp>(val.getLoc(),
                                            builder.getI64IntegerAttr(0)));
      replaceAllUsesInDFTsInRegionWith(val, mrf, opBody);
    }
  }

  // Second generate a CreateAsyncTaskOp that will replace the
  // DataflowTaskOp.  This also includes the necessary handling of
  // operands and results (conversion to/from futures and propagation).
  SmallVector<Value, 4> catOperands;
  int size = 3 + DFTOp.getNumResults() + DFTOp.getNumOperands();
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

  // We need to adjust the results for the CreateAsyncTaskOp which
  // are the work function's returns through pointers passed as
  // parameters.  As this is not supported within MLIR - and mostly
  // unsupported even in the LLVMIR Dialect - this needs to use two
  // placeholders for each output, before and after the
  // CreateAsyncTaskOp.
  IRMapping map;
  for (auto result : DFTOp.getResults()) {
    Type futType = RT::PointerType::get(RT::FutureType::get(result.getType()));
    auto brpp = builder.create<RT::BuildReturnPtrPlaceholderOp>(DFTOp.getLoc(),
                                                                futType);
    map.map(result, brpp->getResult(0));
    catOperands.push_back(brpp->getResult(0));
  }
  for (auto operand : DFTOp.getOperands()) {
    catOperands.push_back(operand);
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

        // If the user is in an scf.forall.in_parallel op, the
        // RT.await_future op must be created in the parent region of
        // the in_parallel op, since scf.forall.in_parallel only
        // allows certain ops in its region.
        if (llvm::dyn_cast_or_null<mlir::scf::InParallelOp>(
                use.getOwner()->getParentOp())) {
          builder.setInsertionPoint(use.getOwner()->getParentOp());
        } else {
          builder.setInsertionPoint(use.getOwner());
        }

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

    module.walk([&](mlir::func::FuncOp func) {
      static int wfn_id = 0;

      // TODO: For now do not attempt to use nested parallelism.
      if (func->getAttr("_dfr_work_function_attribute"))
        return;

      SymbolTable symbolTable = mlir::SymbolTable::getNearestSymbolTable(func);
      SmallVector<std::pair<RT::DataflowTaskOp, func::FuncOp>, 4> outliningMap;

      // Outline DataflowTaskOp bodies to work functions
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

namespace {

// For documentation see Autopar.td
struct StartStopPass : public StartStopBase<StartStopPass> {

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 1> entryPoints;

    // Gather all entry points in the module.
    module.walk([&](mlir::func::FuncOp func) {
      // Work functions are never allowed to be an entry point.
      if (func->getAttr("_dfr_work_function_attribute"))
        return;

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
      // Issue _dfr_start/stop calls for this function
      OpBuilder builder(entryPoint.getBody());
      builder.setInsertionPointToStart(&entryPoint.getBody().front());
      Value useDFRVal = builder.create<arith::ConstantOp>(
          entryPoint.getLoc(), builder.getI64IntegerAttr(1));

      // Check if this entry point uses a context
      Value ctx = nullptr;
      if (dfr::_dfr_is_root_node())
        for (auto arg : llvm::enumerate(entryPoint.getArguments()))
          if (arg.value()
                  .getType()
                  .isa<mlir::concretelang::Concrete::ContextType>()) {
            ctx = arg.value();
            break;
          }
      if (!ctx)
        ctx = builder.create<arith::ConstantOp>(entryPoint.getLoc(),
                                                builder.getI64IntegerAttr(0));

      auto startFunTy = mlir::FunctionType::get(
          entryPoint->getContext(), {useDFRVal.getType(), ctx.getType()}, {});
      (void)insertForwardDeclaration(entryPoint, builder, "_dfr_start",
                                     startFunTy);
      builder.create<mlir::func::CallOp>(entryPoint.getLoc(), "_dfr_start",
                                         mlir::TypeRange(),
                                         mlir::ValueRange({useDFRVal, ctx}));
      builder.setInsertionPoint(entryPoint.getBody().back().getTerminator());
      auto stopFunTy = mlir::FunctionType::get(entryPoint->getContext(),
                                               {useDFRVal.getType()}, {});
      (void)insertForwardDeclaration(entryPoint, builder, "_dfr_stop",
                                     stopFunTy);
      builder.create<mlir::func::CallOp>(entryPoint.getLoc(), "_dfr_stop",
                                         mlir::TypeRange(), useDFRVal);
    }
  }
  StartStopPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // namespace

std::unique_ptr<mlir::Pass> createStartStopPass(bool debug) {
  return std::make_unique<StartStopPass>(debug);
}

namespace {

// For documentation see Autopar.td
struct FinalizeTaskCreationPass
    : public FinalizeTaskCreationBase<FinalizeTaskCreationPass> {

  void runOnOperation() override {
    auto module = getOperation();
    std::vector<Operation *> ops;

    module.walk([&](RT::CreateAsyncTaskOp catOp) {
      OpBuilder builder(catOp);
      SmallVector<Value, 4> operands;

      // Determine if this task needs a runtime context
      Value ctx = nullptr;
      SymbolRefAttr sym =
          catOp->getAttr("workfn").dyn_cast_or_null<SymbolRefAttr>();
      assert(sym && "Work function symbol attribute missing.");
      func::FuncOp workfn = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(catOp, sym));
      assert(workfn && "Task work function missing.");
      if (workfn.getNumArguments() > catOp.getNumOperands() - 3)
        ctx = *catOp->getParentOfType<func::FuncOp>().getArguments().rbegin();
      else
        ctx = builder.create<arith::ConstantOp>(catOp.getLoc(),
                                                builder.getI64IntegerAttr(0));
      int index = 0;
      for (auto op : catOp.getOperands()) {
        operands.push_back(op);
        // Add index in second position - in all cases to avoid
        // checking if needed. It can be null when not relevant.
        if (index == 0)
          operands.push_back(ctx);
        // First three operands are the function pointer, number inputs
        // and number outputs - nothing further to do.
        if (++index <= 3)
          continue;
        auto op_size = getTaskArgumentSizeAndType(op, catOp.getLoc(), builder);
        operands.push_back(op_size.first);
        operands.push_back(op_size.second);
      }

      builder.create<RT::CreateAsyncTaskOp>(catOp.getLoc(), sym, operands);
      ops.push_back(catOp);
    });
    for (auto op : ops) {
      op->erase();
    }

    // If we are building a future on a MemRef, we need to flatten it.

    // TODO: the performance of shared memory can be improved by
    // allowing view-like access instead of cloning, but memory
    // deallocation needs to be synchronized appropriately
    module.walk([&](RT::MakeReadyFutureOp op) {
      OpBuilder builder(op);

      Value val = op.getOperand(0);
      Value clone = op.getOperand(1);
      if (val.getType().isa<mlir::MemRefType>()) {
        MemRefType mrType_base = val.getType().dyn_cast<mlir::MemRefType>();
        MemRefType mrType = mrType_base;
        if (!mrType_base.getLayout().isIdentity()) {
          unsigned rank = mrType_base.getRank();
          mrType = MemRefType::Builder(mrType_base)
                       .setShape(mrType_base.getShape())
                       .setLayout(AffineMapAttr::get(
                           builder.getMultiDimIdentityMap(rank)));
        }

        llvm::SmallVector<mlir::Value> dynamicDimSizes;

        for (auto dimSizeIt : llvm::enumerate(mrType.getShape())) {
          if (mlir::ShapedType::isDynamic(dimSizeIt.value())) {
            mlir::memref::DimOp dimOp = builder.create<mlir::memref::DimOp>(
                val.getLoc(), val, dimSizeIt.index());
            dynamicDimSizes.push_back(dimOp.getResult());
          }
        }

        // We need to make a copy of this MemRef to allow deallocation
        // based on refcounting
        mlir::memref::AllocOp newval = builder.create<mlir::memref::AllocOp>(
            val.getLoc(), mrType, dynamicDimSizes);

        builder.create<mlir::memref::CopyOp>(val.getLoc(), val, newval);
        clone = builder.create<arith::ConstantOp>(op.getLoc(),
                                                  builder.getI64IntegerAttr(1));
        op->setOperand(0, newval);
        op->setOperand(1, clone);
      }
    });

    module.walk([&](RT::WorkFunctionReturnOp op) {
      OpBuilder builder(op);

      Value val = op.getOperand(0);
      if (val.getType().isa<mlir::MemRefType>() &&
          isa<RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
              val.getDefiningOp())) {
        Value newval =
            builder
                .create<mlir::memref::AllocOp>(
                    val.getLoc(), val.getType().dyn_cast<mlir::MemRefType>())
                .getResult();
        builder.create<mlir::memref::CopyOp>(val.getLoc(), val, newval);
        op->setOperand(0, newval);
      }
    });
  }
  FinalizeTaskCreationPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // namespace

std::unique_ptr<mlir::Pass> createFinalizeTaskCreationPass(bool debug) {
  return std::make_unique<FinalizeTaskCreationPass>(debug);
}

namespace {
static void getAliasedUses(Value val, DenseSet<OpOperand *> &aliasedUses) {
  for (auto &use : val.getUses()) {
    aliasedUses.insert(&use);
    if (dyn_cast<ViewLikeOpInterface>(use.getOwner()))
      getAliasedUses(use.getOwner()->getResult(0), aliasedUses);
  }
}

// For documentation see Autopar.td
struct FixupBufferDeallocationPass
    : public FixupBufferDeallocationBase<FixupBufferDeallocationPass> {

  void runOnOperation() override {
    auto module = getOperation();
    std::vector<Operation *> ops;

    module.walk([&](mlir::memref::DeallocOp op) {
      Value alloc = op.getOperand();
      DenseSet<OpOperand *> aliasedUses;
      getAliasedUses(alloc, aliasedUses);

      for (auto use : aliasedUses)
        if (isa<RT::WorkFunctionReturnOp, RT::MakeReadyFutureOp>(
                use->getOwner())) {
          ops.push_back(op);
          return;
        }
    });
    for (auto op : ops) {
      op->erase();
    }

    // For all task return ops, ensure that the return value is
    // allocated - or make a copy otherwise
    module.walk([&](RT::WorkFunctionReturnOp op) {
      Value val = op.getOperand(0);
      Operation *defOp = val.getDefiningOp();
      if (!isa<mlir::memref::AllocOp>(defOp)) {
        assert(val.getType().isa<mlir::MemRefType>());
        OpBuilder builder(op);
        MemRefType mrType = val.getType().dyn_cast<mlir::MemRefType>();
        assert(mrType.getLayout().isIdentity());
        Value newval =
            builder.create<mlir::memref::AllocOp>(val.getLoc(), mrType)
                .getResult();
        builder.create<mlir::memref::CopyOp>(val.getLoc(), val, newval);
        op->setOperand(0, newval);
      }
    });
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
