#ifndef ZAMALANG_CONVERSION_MIDLFHETOLOWLFHE_PATTERNS
#define ZAMALANG_CONVERSION_MIDLFHETOLOWLFHE_PATTERNS

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "zamalang/Dialect/LowLFHE/IR/LowLFHEOps.td"
include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.td"

def createZeroLWEOp : NativeCodeCall<"mlir::zamalang::createZeroLWEOpFromMidLFHE($_builder, $_loc, $0)">;

def ZeroGLWEPattern : Pat<
    (ZeroGLWEOp:$result),
    (createZeroLWEOp $result)>;

def createAddLWEOp : NativeCodeCall<"mlir::zamalang::createLowLFHEOpFromMidLFHE<mlir::zamalang::LowLFHE::AddLweCiphertextsOp>($_builder, $_loc, $0, $1, $2)">;

def AddGLWEPattern : Pat<
    (AddGLWEOp:$result $arg0, $arg1),
    (createAddLWEOp $arg0, $arg1, $result)>;

def createAddPlainLweOp : NativeCodeCall<"mlir::zamalang::createAddPlainLweCiphertext($_builder, $_loc, $0, $1, $2)">;

def AddGLWEIntPattern : Pat<
    (AddGLWEIntOp:$result $arg0, $arg1),
    (createAddPlainLweOp $arg0, $arg1, $result)>;

def createMulClearLweOp : NativeCodeCall<"mlir::zamalang::createMulClearLweCiphertext($_builder, $_loc, $0, $1, $2)">;

def MulGLWEIntPattern : Pat<
    (MulGLWEIntOp:$result $arg0, $arg1),
    (createMulClearLweOp $arg0, $arg1, $result)>;

def createSubIntLweOp : NativeCodeCall<"mlir::zamalang::createSubIntLweCiphertext($_builder, $_loc, $0, $1, $2)">;

def SubIntGLWEPattern : Pat<
    (SubIntGLWEOp:$result $arg0, $arg1),
    (createSubIntLweOp $arg0, $arg1, $result)>;

#endif
