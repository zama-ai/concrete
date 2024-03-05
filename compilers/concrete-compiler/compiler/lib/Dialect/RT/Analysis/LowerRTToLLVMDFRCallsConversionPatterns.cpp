// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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
#include <llvm/Support/Compiler.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/VectorPattern.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
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

mlir::Type getVoidPtrI64Type(ConversionPatternRewriter &rewriter) {
  return mlir::LLVM::LLVMPointerType::get(
      mlir::IntegerType::get(rewriter.getContext(), 64));
}

LLVM::LLVMFuncOp getOrInsertFuncOpDecl(mlir::Operation *op,
                                       llvm::StringRef funcName,
                                       LLVM::LLVMFunctionType funcType,
                                       ConversionPatternRewriter &rewriter) {
  // Check if the function is already in the symbol table
  auto module = op->getParentOfType<ModuleOp>();
  auto funcOp = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
  if (!funcOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    funcOp =
        rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), funcName, funcType);
    funcOp.setPrivate();
  } else {
    if (!funcOp.isPrivate()) {
      op->emitError()
          << "the function \"" << funcName
          << "\" conflicts with the Dataflow Runtime API, please rename.";
      return nullptr;
    }
  }
  return funcOp;
}

/// This function is only needed for debug purposes to inspect values
/// in the generated code - it is therefore not generally in use.
LLVM_ATTRIBUTE_UNUSED void
insertPrintDebugCall(ConversionPatternRewriter &rewriter, mlir::Operation *op,
                     Value val) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto printFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(rewriter.getContext()), {}, /*isVariadic=*/true);
  auto printFnOp =
      getOrInsertFuncOpDecl(op, "_dfr_print_debug", printFnType, rewriter);
  rewriter.create<LLVM::CallOp>(op->getLoc(), printFnOp, val);
}

struct MakeReadyFutureOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::MakeReadyFutureOp> {
  using ConvertOpToLLVMPattern<RT::MakeReadyFutureOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::MakeReadyFutureOp mrfOp,
                  RT::MakeReadyFutureOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);

    // Normally this function takes a pointer as parameter
    auto mrfFuncType = LLVM::LLVMFunctionType::get(getVoidPtrI64Type(rewriter),
                                                   {}, /*isVariadic=*/true);
    auto mrfFuncOp = getOrInsertFuncOpDecl(mrfOp, "_dfr_make_ready_future",
                                           mrfFuncType, rewriter);

    // In order to support non pointer types, we need to allocate
    // explicitly space that we can reference as a base for the
    // future.
    auto allocFuncOp = mlir::LLVM::lookupOrCreateMallocFn(
        mrfOp->getParentOfType<ModuleOp>(), getIndexType(),
        getTypeConverter()->useOpaquePointers());
    auto sizeBytes = getSizeInBytes(
        mrfOp.getLoc(), adaptor.getOperands().getTypes().front(), rewriter);

    auto results =
        rewriter.create<LLVM::CallOp>(mrfOp.getLoc(), allocFuncOp, sizeBytes);
    Value allocatedPtr = rewriter.create<mlir::LLVM::BitcastOp>(
        mrfOp.getLoc(),
        mlir::LLVM::LLVMPointerType::get(
            adaptor.getOperands().getTypes().front()),
        results.getResult());
    rewriter.create<LLVM::StoreOp>(mrfOp.getLoc(),
                                   adaptor.getOperands().front(), allocatedPtr);
    SmallVector<Value, 4> mrfOperands = {adaptor.getOperands()};
    mrfOperands[0] = allocatedPtr;
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(mrfOp, mrfFuncOp, mrfOperands);
    return mlir::success();
  }
};
struct AwaitFutureOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::AwaitFutureOp> {
  using ConvertOpToLLVMPattern<RT::AwaitFutureOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::AwaitFutureOp afOp, RT::AwaitFutureOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    auto afFuncType = LLVM::LLVMFunctionType::get(
        mlir::LLVM::LLVMPointerType::get(getVoidPtrI64Type(rewriter)),
        {getVoidPtrI64Type(rewriter)});
    auto afFuncOp =
        getOrInsertFuncOpDecl(afOp, "_dfr_await_future", afFuncType, rewriter);
    auto afCallOp = rewriter.create<LLVM::CallOp>(afOp.getLoc(), afFuncOp,
                                                  adaptor.getOperands());
    Value futVal = rewriter.create<mlir::LLVM::BitcastOp>(
        afOp.getLoc(),
        mlir::LLVM::LLVMPointerType::get(
            (*getTypeConverter()).convertType(afOp.getResult().getType())),
        afCallOp.getResult());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(afOp, futVal);
    return success();
  }
};
struct CreateAsyncTaskOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::CreateAsyncTaskOp> {
  using ConvertOpToLLVMPattern<RT::CreateAsyncTaskOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::CreateAsyncTaskOp catOp,
                  RT::CreateAsyncTaskOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto catFuncType =
        LLVM::LLVMFunctionType::get(getVoidType(), {}, /*isVariadic=*/true);
    auto catFuncOp = getOrInsertFuncOpDecl(catOp, "_dfr_create_async_task",
                                           catFuncType, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(catOp, catFuncOp,
                                              adaptor.getOperands());
    return success();
  }
};
struct RegisterTaskWorkFunctionOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::RegisterTaskWorkFunctionOp> {
  using ConvertOpToLLVMPattern<
      RT::RegisterTaskWorkFunctionOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::RegisterTaskWorkFunctionOp rtwfOp,
                  RT::RegisterTaskWorkFunctionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rtwfFuncType =
        LLVM::LLVMFunctionType::get(getVoidType(), {}, /*isVariadic=*/true);
    auto rtwfFuncOp = getOrInsertFuncOpDecl(
        rtwfOp, "_dfr_register_work_function", rtwfFuncType, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(rtwfOp, rtwfFuncOp,
                                              adaptor.getOperands());
    return success();
  }
};
struct DeallocateFutureOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::DeallocateFutureOp> {
  using ConvertOpToLLVMPattern<RT::DeallocateFutureOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::DeallocateFutureOp dfOp,
                  RT::DeallocateFutureOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dfFuncType = LLVM::LLVMFunctionType::get(
        getVoidType(), {getVoidPtrI64Type(rewriter)});
    auto dfFuncOp = getOrInsertFuncOpDecl(dfOp, "_dfr_deallocate_future",
                                          dfFuncType, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(dfOp, dfFuncOp,
                                              adaptor.getOperands());
    return success();
  }
};
struct DeallocateFutureDataOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::DeallocateFutureDataOp> {
  using ConvertOpToLLVMPattern<
      RT::DeallocateFutureDataOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::DeallocateFutureDataOp dfdOp,
                  RT::DeallocateFutureDataOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dfdFuncType = LLVM::LLVMFunctionType::get(
        getVoidType(), {getVoidPtrI64Type(rewriter)});
    auto dfdFuncOp = getOrInsertFuncOpDecl(dfdOp, "_dfr_deallocate_future_data",
                                           dfdFuncType, rewriter);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(dfdOp, dfdFuncOp,
                                              adaptor.getOperands());
    return success();
  }
};
struct BuildReturnPtrPlaceholderOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::BuildReturnPtrPlaceholderOp> {
  using ConvertOpToLLVMPattern<
      RT::BuildReturnPtrPlaceholderOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::BuildReturnPtrPlaceholderOp befOp,
                  RT::BuildReturnPtrPlaceholderOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);

    // BuildReturnPtrPlaceholder is a placeholder for generating a memory
    // location where a pointer to allocated memory can be written so
    // that we can return outputs from task work function.
    Value one = rewriter.create<arith::ConstantOp>(
        befOp.getLoc(),
        (*getTypeConverter()).convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(
            (*getTypeConverter()).convertType(rewriter.getIndexType()), 1));
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        befOp, mlir::LLVM::LLVMPointerType::get(getVoidPtrI64Type(rewriter)),
        one, 0);
    return success();
  }
};
struct DerefReturnPtrPlaceholderOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::DerefReturnPtrPlaceholderOp> {
  using ConvertOpToLLVMPattern<
      RT::DerefReturnPtrPlaceholderOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::DerefReturnPtrPlaceholderOp drppOp,
                  RT::DerefReturnPtrPlaceholderOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // DerefReturnPtrPlaceholder is a placeholder for generating a
    // dereference operation for the pointer used to get results from
    // task.
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(drppOp,
                                              adaptor.getOperands().front());
    return success();
  }
};
struct DerefWorkFunctionArgumentPtrPlaceholderOpInterfaceLowering
    : public ConvertOpToLLVMPattern<
          RT::DerefWorkFunctionArgumentPtrPlaceholderOp> {
  using ConvertOpToLLVMPattern<
      RT::DerefWorkFunctionArgumentPtrPlaceholderOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      RT::DerefWorkFunctionArgumentPtrPlaceholderOp dwfappOp,
      RT::DerefWorkFunctionArgumentPtrPlaceholderOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);

    // DerefWorkFunctionArgumentPtrPlaceholderOp is a placeholder for
    // generating a dereference operation for the pointer used to pass
    // arguments to the task.
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(dwfappOp,
                                              adaptor.getOperands().front());
    return success();
  }
};
struct WorkFunctionReturnOpInterfaceLowering
    : public ConvertOpToLLVMPattern<RT::WorkFunctionReturnOp> {
  using ConvertOpToLLVMPattern<
      RT::WorkFunctionReturnOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(RT::WorkFunctionReturnOp wfrOp,
                  RT::WorkFunctionReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
        wfrOp, adaptor.getOperands().front(), adaptor.getOperands().back());
    return success();
  }
};
} // end anonymous namespace
} // namespace concretelang
} // namespace mlir

void mlir::concretelang::populateRTToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    MakeReadyFutureOpInterfaceLowering,
    AwaitFutureOpInterfaceLowering,
    BuildReturnPtrPlaceholderOpInterfaceLowering,
    DerefReturnPtrPlaceholderOpInterfaceLowering,
    DerefWorkFunctionArgumentPtrPlaceholderOpInterfaceLowering,
    CreateAsyncTaskOpInterfaceLowering,
    RegisterTaskWorkFunctionOpInterfaceLowering,
    DeallocateFutureOpInterfaceLowering,
    DeallocateFutureDataOpInterfaceLowering,
    WorkFunctionReturnOpInterfaceLowering>(converter);
  // clang-format on
}
