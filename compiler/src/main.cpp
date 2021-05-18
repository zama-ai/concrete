#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>

#include "llvm/Support/SourceMgr.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

mlir::FuncOp buildFunction(mlir::OpBuilder &builder) {
    mlir::FunctionType func_type = builder.getFunctionType({ mlir::zamalang::HLFHE::EncryptedIntegerType::get(builder.getContext(), 32) }, llvm::None);

    mlir::FuncOp funcOp =
        mlir::FuncOp::create(builder.getUnknownLoc(), "hlfhe", func_type);

    mlir::FuncOp function(funcOp);
    mlir::Block &entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value v1 = builder.create<mlir::ConstantFloatOp>(
        builder.getUnknownLoc(),
        llvm::APFloat(llvm::APFloat::IEEEsingle(), "1.0"),
        builder.getF32Type());

    // TODO: create v2 as EncryptedInteger and add it with v1

    // mlir::Value v2 =
    //     builder.create<mlir::zamalang::HLFHE::EncryptedIntegerType>(
    //         builder.getUnknownLoc());

    mlir::Value c1 = builder.create<mlir::zamalang::HLFHE::AddEintIntOp>(
        builder.getUnknownLoc(), v1, v1);

    builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

    return funcOp;
}

int main(int argc, char **argv) {
    mlir::MLIRContext context;

    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();

    mlir::OpBuilder builder(&context);

    mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

    module.push_back(buildFunction(builder));

    module.dump();

    return 0;
}
