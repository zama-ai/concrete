#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHE_TYPES
#define CONCRETELANG_DIALECT_TFHE_IR_TFHE_TYPES

// TODO: MLWE / GSW

include "concretelang/Dialect/TFHE/IR/TFHEDialect.td"
include "mlir/IR/BuiltinTypes.td"

class TFHE_Type<string name, list<Trait> traits = []> : TypeDef<TFHE_Dialect, name, traits> { }

def GLWECipherTextType : TFHE_Type<"GLWECipherText", [MemRefElementTypeInterface]> {
    let mnemonic = "glwe";

    let summary = "A GLWE ciphertext";

    let description = [{
        An GLWE cipher text
    }];

    let parameters = (ins
        // The size of the mask
        "signed":$dimension,
        // Size of the polynome
        "signed":$polynomialSize,
        // Number of bits of the ciphertext
        "signed":$bits,
        // Number of bits of the plain text representation
        "signed":$p
    );

    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = true;
}

#endif
