#ifndef CONCRETELANG_CONVERSION_FHETOTFHE_PATTERNS
#define CONCRETELANG_CONVERSION_FHETOTFHE_PATTERNS

include "mlir/Pass/PassBase.td"
include "mlir/IR/PatternBase.td"
include "concretelang/Dialect/FHE/IR/FHEOps.td"
include "concretelang/Dialect/TFHE/IR/TFHEOps.td"

def createZeroGLWEOp : NativeCodeCall<"mlir::concretelang::createZeroGLWEOpFromFHE($_builder, $_loc, $0)">;

def ZeroEintPattern : Pat<
    (ZeroEintOp:$result),
    (createZeroGLWEOp $result)>;

def createAddGLWEIntOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromFHE<mlir::concretelang::TFHE::AddGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def AddEintIntPattern : Pat<
    (AddEintIntOp:$result $arg0, $arg1),
    (createAddGLWEIntOp $arg0, $arg1, $result)>;

def createAddGLWEOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromFHE<mlir::concretelang::TFHE::AddGLWEOp>($_builder, $_loc, $0, $1, $2)">;

def AddEintPattern : Pat<
    (AddEintOp:$result $arg0, $arg1),
    (createAddGLWEOp $arg0, $arg1, $result)>;

def createSubGLWEIntOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromFHE<mlir::concretelang::TFHE::SubGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def SubIntEintPattern : Pat<
    (SubIntEintOp:$result $arg0, $arg1),
    (createSubGLWEIntOp $arg0, $arg1, $result)>;

def createNegGLWEOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromFHE<mlir::concretelang::TFHE::NegGLWEOp>($_builder, $_loc, $0, $1)">;

def NegEintPattern : Pat<
    (NegEintOp:$result $arg0),
    (createNegGLWEOp $arg0, $result)>;

def createMulGLWEIntOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromFHE<mlir::concretelang::TFHE::MulGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def MulEintIntPattern : Pat<
    (MulEintIntOp:$result $arg0, $arg1),
    (createMulGLWEIntOp $arg0, $arg1, $result)>;

#endif
