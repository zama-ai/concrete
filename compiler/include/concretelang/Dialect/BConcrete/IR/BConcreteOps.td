#ifndef ZAMALANG_DIALECT_BConcrete_IR_BConcrete_OPS
#define ZAMALANG_DIALECT_BConcrete_IR_BConcrete_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"

include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"

class BConcrete_Op<string mnemonic, list<Trait> traits = []> :
    Op<BConcrete_Dialect, mnemonic, traits>;

def AddLweBuffersOp : BConcrete_Op<"add_lwe_buffer"> {
    let arguments = (ins  
        1DTensorOf<[I64]>:$lhs,
        1DTensorOf<[I64]>:$rhs
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def AddPlaintextLweBufferOp : BConcrete_Op<"add_plaintext_lwe_buffer"> {
    let arguments = (ins 1DTensorOf<[I64]>:$lhs, I64:$rhs);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def MulCleartextLweBufferOp : BConcrete_Op<"mul_cleartext_lwe_buffer"> {
    let arguments = (ins 1DTensorOf<[I64]>:$lhs, I64:$rhs);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def NegateLweBufferOp : BConcrete_Op<"negate_lwe_buffer"> {
    let arguments = (ins 1DTensorOf<[I64]>:$ciphertext);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def FillGlweFromTable : BConcrete_Op<"fill_glwe_from_table"> {
    let arguments = (ins
        1DTensorOf<[I64]>:$glwe,
        I32Attr:$glweDimension,
        I32Attr:$polynomialSize,
        I32Attr:$outPrecision,
        1DTensorOf<[I64]>:$table
    );
    let results = (outs);
}

def KeySwitchLweBufferOp : BConcrete_Op<"keyswitch_lwe_buffer"> {
    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        1DTensorOf<[I64]>:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BootstrapLweBufferOp : BConcrete_Op<"bootstrap_lwe_buffer"> {
    let arguments = (ins
        // LweBootstrapKeyType:$bootstrap_key,
        1DTensorOf<[I64]>:$input_ciphertext,
        1DTensorOf<[I64]>:$accumulator,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}


#endif
