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

def BConcrete_AddLweBuffersOp : BConcrete_Op<"add_lwe_buffer"> {
    let arguments = (ins  
        1DTensorOf<[I64]>:$lhs,
        1DTensorOf<[I64]>:$rhs
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_AddCRTLweBuffersOp : BConcrete_Op<"add_crt_lwe_buffer"> {
    let arguments = (ins
        2DTensorOf<[I64]>:$lhs,
        2DTensorOf<[I64]>:$rhs,
        I64ArrayAttr:$crtDecomposition
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

def BConcrete_AddPlaintextLweBufferOp : BConcrete_Op<"add_plaintext_lwe_buffer"> {
    let arguments = (ins 1DTensorOf<[I64]>:$lhs, I64:$rhs);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_AddPlaintextCRTLweBufferOp : BConcrete_Op<"add_plaintext_crt_lwe_buffer"> {
    let arguments = (ins
        2DTensorOf<[I64]>:$lhs,
        AnyInteger:$rhs,
        I64ArrayAttr:$crtDecomposition
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

def BConcrete_MulCleartextLweBufferOp : BConcrete_Op<"mul_cleartext_lwe_buffer"> {
    let arguments = (ins 1DTensorOf<[I64]>:$lhs, I64:$rhs);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_MulCleartextCRTLweBufferOp : BConcrete_Op<"mul_cleartext_crt_lwe_buffer"> {
    let arguments = (ins
        2DTensorOf<[I64]>:$lhs,
        AnyInteger:$rhs,
        I64ArrayAttr:$crtDecomposition
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

def BConcrete_NegateLweBufferOp : BConcrete_Op<"negate_lwe_buffer"> {
    let arguments = (ins 1DTensorOf<[I64]>:$ciphertext);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_NegateCRTLweBufferOp : BConcrete_Op<"negate_crt_lwe_buffer"> {
    let arguments = (ins
        2DTensorOf<[I64]>:$ciphertext,
        I64ArrayAttr:$crtDecomposition
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

def BConcrete_FillGlweFromTable : BConcrete_Op<"fill_glwe_from_table"> {
    let arguments = (ins
        1DTensorOf<[I64]>:$glwe,
        I32Attr:$glweDimension,
        I32Attr:$polynomialSize,
        I32Attr:$outPrecision,
        1DTensorOf<[I64]>:$table
    );
    let results = (outs);
}

def BConcrete_KeySwitchLweBufferOp : BConcrete_Op<"keyswitch_lwe_buffer"> {
    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        1DTensorOf<[I64]>:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_BootstrapLweBufferOp : BConcrete_Op<"bootstrap_lwe_buffer"> {
    let arguments = (ins
        // LweBootstrapKeyType:$bootstrap_key,
        1DTensorOf<[I64]>:$input_ciphertext,
        1DTensorOf<[I64]>:$accumulator,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

// TODO(16bits): hack
def BConcrete_WopPBSCRTLweBufferOp : BConcrete_Op<"wop_pbs_crt_lwe_buffer"> {
    let arguments = (ins
        2DTensorOf<[I64]>:$ciphertext,
        1DTensorOf<[I64]>:$lookupTable,
        // Bootstrap parameters
        I32Attr : $bootstrapLevel,
        I32Attr : $bootstrapBaseLog,
        // Keyswitch parameters
        I32Attr : $keyswitchLevel,
        I32Attr : $keyswitchBaseLog,
        // Packing keyswitch key parameters
        I32Attr : $packingKeySwitchInputLweDimension,
        I32Attr : $packingKeySwitchinputLweCount,
        I32Attr : $packingKeySwitchoutputPolynomialSize,
        I32Attr : $packingKeySwitchLevel,
        I32Attr : $packingKeySwitchBaseLog,
        // Circuit bootstrap parameters
        I32Attr : $circuitBootstrapLevel,
        I32Attr : $circuitBootstrapBaseLog
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

#endif
