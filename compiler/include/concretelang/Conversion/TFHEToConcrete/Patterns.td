#ifndef CONCRETELANG_CONVERSION_TFHETOCONCRETE_PATTERNS
#define CONCRETELANG_CONVERSION_TFHETOCONCRETE_PATTERNS

include "mlir/Pass/PassBase.td"
include "mlir/IR/PatternBase.td"
include "concretelang/Dialect/Concrete/IR/ConcreteOps.td"
include "concretelang/Dialect/TFHE/IR/TFHEOps.td"

def createZeroLWEOp : NativeCodeCall<"mlir::concretelang::createZeroLWEOpFromTFHE($_builder, $_loc, $0)">;

def ZeroGLWEPattern : Pat<
    (ZeroGLWEOp:$result),
    (createZeroLWEOp $result)>;

def createAddLWEOp : NativeCodeCall<"mlir::concretelang::createConcreteOpFromTFHE<mlir::concretelang::Concrete::AddLweCiphertextsOp>($_builder, $_loc, $0, $1, $2)">;

def AddGLWEPattern : Pat<
    (AddGLWEOp:$result $arg0, $arg1),
    (createAddLWEOp $arg0, $arg1, $result)>;

def createAddPlainLweOp : NativeCodeCall<"mlir::concretelang::createAddPlainLweCiphertext($_builder, $_loc, $0, $1, $2)">;

def AddGLWEIntPattern : Pat<
    (AddGLWEIntOp:$result $arg0, $arg1),
    (createAddPlainLweOp $arg0, $arg1, $result)>;

def createMulClearLweOp : NativeCodeCall<"mlir::concretelang::createMulClearLweCiphertext($_builder, $_loc, $0, $1, $2)">;

def MulGLWEIntPattern : Pat<
    (MulGLWEIntOp:$result $arg0, $arg1),
    (createMulClearLweOp $arg0, $arg1, $result)>;

def createSubIntLweOp : NativeCodeCall<"mlir::concretelang::createSubIntLweCiphertext($_builder, $_loc, $0, $1, $2)">;

def SubGLWEIntPattern : Pat<
    (SubGLWEIntOp:$result $arg0, $arg1),
    (createSubIntLweOp $arg0, $arg1, $result)>;

def createNegLweOp : NativeCodeCall<"mlir::concretelang::createNegLweCiphertext($_builder, $_loc, $0, $1)">;

def NegGLWEPattern : Pat<
    (NegGLWEOp:$result $arg0),
    (createNegLweOp $arg0, $result)>;

#endif
