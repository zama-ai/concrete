#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"

class Concrete_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Concrete_Dialect, mnemonic, traits>;

def ZeroLWEOp : Concrete_Op<"zero"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs LweCiphertextType:$out);
}

def AddLweCiphertextsOp : Concrete_Op<"add_lwe_ciphertexts"> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins LweCiphertextType:$lhs, LweCiphertextType:$rhs);
    let results = (outs LweCiphertextType:$result);
}

def AddPlaintextLweCiphertextOp : Concrete_Op<"add_plaintext_lwe_ciphertext"> {
    let summary = "Returns the sum of a clear integer and a lwe ciphertext";

    let arguments = (ins LweCiphertextType:$lhs, PlaintextType:$rhs);
    let results = (outs LweCiphertextType:$result);
}

def MulCleartextLweCiphertextOp : Concrete_Op<"mul_cleartext_lwe_ciphertext"> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins LweCiphertextType:$lhs, CleartextType:$rhs);
    let results = (outs LweCiphertextType:$result);
}

def NegateLweCiphertextOp : Concrete_Op<"negate_lwe_ciphertext"> {
    let summary = "Negates a lwe ciphertext";

    let arguments = (ins LweCiphertextType:$ciphertext);
    let results = (outs LweCiphertextType:$result);
}

def GlweFromTable : Concrete_Op<"glwe_from_table"> {
    let summary = "Creates a GLWE ciphertext which is the trivial encrytion of a the input table interpreted as a polynomial (to use later in a bootstrap)";

    let arguments = (ins TensorOf<[AnyInteger]>:$table, I32Attr:$polynomialSize, I32Attr:$glweDimension, I32Attr:$p);
    let results = (outs GlweCiphertextType:$result);
}

def BootstrapLweOp : Concrete_Op<"bootstrap_lwe"> {
    let summary = "Bootstraps a LWE ciphertext with a GLWE trivial encryption of the lookup table";


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

def KeySwitchLweOp : Concrete_Op<"keyswitch_lwe"> {
    let summary = "Keyswitches a LWE ciphertext";

    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        LweCiphertextType:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs LweCiphertextType:$result);
}

def EncodeIntOp : Concrete_Op<"encode_int"> {
    let summary = "Encodes an integer (for it to later be added to a LWE ciphertext)";

    let arguments = (ins AnyInteger:$i);
    let results = (outs PlaintextType:$plaintext);
}

def IntToCleartextOp : Concrete_Op<"int_to_cleartext"> {
    let summary = "Keyswitches a LWE ciphertext";

    let arguments = (ins AnyInteger:$i);
    let results = (outs CleartextType:$cleartext);
}

#endif
