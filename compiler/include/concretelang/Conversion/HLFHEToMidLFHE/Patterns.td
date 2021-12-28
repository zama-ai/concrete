#ifndef CONCRETELANG_CONVERSION_HLFHETOMIDLFHE_PATTERNS
#define CONCRETELANG_CONVERSION_HLFHETOMIDLFHE_PATTERNS

include "mlir/Pass/PassBase.td"
include "concretelang/Dialect/HLFHE/IR/HLFHEOps.td"
include "concretelang/Dialect/MidLFHE/IR/MidLFHEOps.td"

def createZeroGLWEOp : NativeCodeCall<"mlir::concretelang::createZeroGLWEOpFromHLFHE($_builder, $_loc, $0)">;

def ZeroEintPattern : Pat<
    (ZeroEintOp:$result),
    (createZeroGLWEOp $result)>;

def createAddGLWEIntOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromHLFHE<mlir::concretelang::MidLFHE::AddGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def AddEintIntPattern : Pat<
    (AddEintIntOp:$result $arg0, $arg1),
    (createAddGLWEIntOp $arg0, $arg1, $result)>;

def createAddGLWEOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromHLFHE<mlir::concretelang::MidLFHE::AddGLWEOp>($_builder, $_loc, $0, $1, $2)">;

def AddEintPattern : Pat<
    (AddEintOp:$result $arg0, $arg1),
    (createAddGLWEOp $arg0, $arg1, $result)>;

def createSubIntGLWEOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromHLFHE<mlir::concretelang::MidLFHE::SubIntGLWEOp>($_builder, $_loc, $0, $1, $2)">;

def SubIntEintPattern : Pat<
    (SubIntEintOp:$result $arg0, $arg1),
    (createSubIntGLWEOp $arg0, $arg1, $result)>;

def createNegGLWEOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromHLFHE<mlir::concretelang::MidLFHE::NegGLWEOp>($_builder, $_loc, $0, $1)">;

def NegEintPattern : Pat<
    (NegEintOp:$result $arg0),
    (createNegGLWEOp $arg0, $result)>;

def createMulGLWEIntOp : NativeCodeCall<"mlir::concretelang::createGLWEOpFromHLFHE<mlir::concretelang::MidLFHE::MulGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def MulEintIntPattern : Pat<
    (MulEintIntOp:$result $arg0, $arg1),
    (createMulGLWEIntOp $arg0, $arg1, $result)>;

def createApplyLookupTableGLWEOp : NativeCodeCall<"mlir::concretelang::createApplyLookupTableGLWEOpFromHLFHE($_builder, $_loc, $0, $1, $2)">;

def ApplyLookupTableEintPattern : Pat<
    (ApplyLookupTableEintOp:$result $arg0, $arg1),
    (createApplyLookupTableGLWEOp $arg0, $arg1, $result)>;

#endif
