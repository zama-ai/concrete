#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"

class Concrete_Op<string mnemonic, list<Trait> traits = []> :
    Op<Concrete_Dialect, mnemonic, traits>;

def Concrete_ZeroLWEOp : Concrete_Op<"zero"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs Concrete_LweCiphertextType:$out);
}

def Concrete_ZeroTensorLWEOp : Concrete_Op<"zero_tensor"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs Type<And<[TensorOf<[Concrete_LweCiphertextType]>.predicate, HasStaticShapePred]>>:$tensor);
}

def Concrete_AddLweCiphertextsOp : Concrete_Op<"add_lwe_ciphertexts"> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins Concrete_LweCiphertextType:$lhs, Concrete_LweCiphertextType:$rhs);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_AddPlaintextLweCiphertextOp : Concrete_Op<"add_plaintext_lwe_ciphertext"> {
    let summary = "Returns the sum of a clear integer and a lwe ciphertext";

    let arguments = (ins Concrete_LweCiphertextType:$lhs, AnyInteger:$rhs);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_MulCleartextLweCiphertextOp : Concrete_Op<"mul_cleartext_lwe_ciphertext"> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins Concrete_LweCiphertextType:$lhs, AnyInteger:$rhs);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_NegateLweCiphertextOp : Concrete_Op<"negate_lwe_ciphertext"> {
    let summary = "Negates a lwe ciphertext";

    let arguments = (ins Concrete_LweCiphertextType:$ciphertext);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_GlweFromTable : Concrete_Op<"glwe_from_table"> {
    let summary = "Creates a GLWE ciphertext which is the trivial encrytion of a the input table interpreted as a polynomial (to use later in a bootstrap)";

    let arguments = (ins 1DTensorOf<[I64]>:$table);
    let results = (outs Concrete_GlweCiphertextType:$result);
}

def Concrete_BootstrapLweOp : Concrete_Op<"bootstrap_lwe"> {
    let summary = "Bootstraps a LWE ciphertext with a GLWE trivial encryption of the lookup table";

    let arguments = (ins 
        Concrete_LweCiphertextType:$input_ciphertext,
        Concrete_GlweCiphertextType:$accumulator,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_KeySwitchLweOp : Concrete_Op<"keyswitch_lwe"> {
    let summary = "Keyswitches a LWE ciphertext";

    let arguments = (ins
        Concrete_LweCiphertextType:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs Concrete_LweCiphertextType:$result);
}

// TODO(16bits): hack
def Concrete_WopPBSLweOp : Concrete_Op<"wop_pbs_lwe"> {
    let summary = "";

    let arguments = (ins
        Concrete_LweCiphertextType:$ciphertext,
        1DTensorOf<[I64]>:$accumulator,
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
    let results = (outs Concrete_LweCiphertextType:$result);
}

#endif
