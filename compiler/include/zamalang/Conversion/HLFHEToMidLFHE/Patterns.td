#ifndef ZAMALANG_CONVERSION_HLFHETOMIDLFHE_PATTERNS
#define ZAMALANG_CONVERSION_HLFHETOMIDLFHE_PATTERNS

include "zamalang/Dialect/HLFHE/IR/HLFHEOps.td"
include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.td"

def createAddGLWEIntOp : NativeCodeCall<"mlir::zamalang::createGLWEOpFromHLFHE<mlir::zamalang::MidLFHE::AddGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def AddEintIntPattern : Pat<
    (AddEintIntOp:$result $arg0, $arg1),
    (createAddGLWEIntOp $arg0, $arg1, $result)>;

def createAddGLWEOp : NativeCodeCall<"mlir::zamalang::createGLWEOpFromHLFHE<mlir::zamalang::MidLFHE::AddGLWEOp>($_builder, $_loc, $0, $1, $2)">;

def AddEintPattern : Pat<
    (AddEintOp:$result $arg0, $arg1),
    (createAddGLWEOp $arg0, $arg1, $result)>;

def createSubIntGLWEOp : NativeCodeCall<"mlir::zamalang::createGLWEOpFromHLFHE<mlir::zamalang::MidLFHE::SubIntGLWEOp>($_builder, $_loc, $0, $1, $2)">;

def SubIntEintPattern : Pat<
    (SubIntEintOp:$result $arg0, $arg1),
    (createSubIntGLWEOp $arg0, $arg1, $result)>;

def createMulGLWEIntOp : NativeCodeCall<"mlir::zamalang::createGLWEOpFromHLFHE<mlir::zamalang::MidLFHE::MulGLWEIntOp>($_builder, $_loc, $0, $1, $2)">;

def MulEintIntPattern : Pat<
    (MulEintIntOp:$result $arg0, $arg1),
    (createMulGLWEIntOp $arg0, $arg1, $result)>;

def createApplyLookupTableGLWEOp : NativeCodeCall<"mlir::zamalang::createApplyLookupTableGLWEOpFromHLFHE($_builder, $_loc, $0, $1, $2)">;

def ApplyLookupTableEintPattern : Pat<
    (ApplyLookupTableEintOp:$result $arg0, $arg1),
    (createApplyLookupTableGLWEOp $arg0, $arg1, $result)>;

#endif
