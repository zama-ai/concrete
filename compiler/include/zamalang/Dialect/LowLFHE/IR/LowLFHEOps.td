#ifndef ZAMALANG_DIALECT_LowLFHE_IR_LowLFHE_OPS
#define ZAMALANG_DIALECT_LowLFHE_IR_LowLFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.td"
include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.td"

class LowLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<LowLFHE_Dialect, mnemonic, traits>;

def ZeroLWEOp : LowLFHE_Op<"zero"> {
    let arguments = (ins);
    let results = (outs LweCiphertextType:$out);
}

def AddLweCiphertextsOp : LowLFHE_Op<"add_lwe_ciphertexts"> {
    let arguments = (ins LweCiphertextType:$lhs, LweCiphertextType:$rhs);
    let results = (outs LweCiphertextType:$result);
}

def AddPlaintextLweCiphertextOp : LowLFHE_Op<"add_plaintext_lwe_ciphertext"> {
    let arguments = (ins LweCiphertextType:$lhs, PlaintextType:$rhs);
    let results = (outs LweCiphertextType:$result);
}

def MulCleartextLweCiphertextOp : LowLFHE_Op<"mul_cleartext_lwe_ciphertext"> {
    let arguments = (ins LweCiphertextType:$lhs, CleartextType:$rhs);
    let results = (outs LweCiphertextType:$result);
}

def NegateLweCiphertextOp : LowLFHE_Op<"negate_lwe_ciphertext"> {
    let arguments = (ins LweCiphertextType:$ciphertext);
    let results = (outs LweCiphertextType:$result);
}

def GlweFromTable : LowLFHE_Op<"glwe_from_table"> {
    let arguments = (ins TensorOf<[AnyInteger]>:$table, I32Attr:$polynomialSize, I32Attr:$glweDimension, I32Attr:$p);
    let results = (outs GlweCiphertextType:$result);
}

def BootstrapLweOp : LowLFHE_Op<"bootstrap_lwe"> {
    let arguments = (ins 
        // LweBootstrapKeyType:$bootstrap_key,
        LweCiphertextType:$input_ciphertext,
        GlweCiphertextType:$accumulator,
        I32Attr:$glweDimension,
        I32Attr:$polynomialSize,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs LweCiphertextType:$result);
}

def KeySwitchLweOp : LowLFHE_Op<"keyswitch_lwe"> {
    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        LweCiphertextType:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs LweCiphertextType:$result);
}

def EncodeIntOp : LowLFHE_Op<"encode_int"> {
    let arguments = (ins AnyInteger:$i);
    let results = (outs PlaintextType:$plaintext);
}

def IntToCleartextOp : LowLFHE_Op<"int_to_cleartext"> {
    let arguments = (ins AnyInteger:$i);
    let results = (outs CleartextType:$cleartext);
}

#endif
