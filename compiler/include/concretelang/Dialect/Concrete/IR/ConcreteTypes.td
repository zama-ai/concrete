#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_TYPES
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_TYPES

include "mlir/IR/BuiltinTypes.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"

class Concrete_Type<string name, list<Trait> traits = []> : TypeDef<Concrete_Dialect, name, traits> { }

def Concrete_GlweCiphertextType : Concrete_Type<"GlweCiphertext"> {
    let mnemonic = "glwe_ciphertext";

    let summary = "A GLWE ciphertext (encryption of a polynomial of fixed-precision integers)";

    let description = [{
       GLWE ciphertext.
    }];

    let hasCustomAssemblyFormat = 1;

    let parameters = (ins
        "signed":$glweDimension,
        "signed":$polynomialSize,
        // Precision of the lwe ciphertext
        "signed":$p
    );
}

def Concrete_LweCiphertextType : Concrete_Type<"LweCiphertext", [MemRefElementTypeInterface]> {
    let mnemonic = "lwe_ciphertext";

    let summary = "A LWE ciphertext (encryption of a fixed-precision integer)";

    let description = [{
       Learning With Error ciphertext.
    }];


    let parameters = (ins
        // The dimension of the lwe ciphertext
        "signed":$dimension,
        // Precision of the lwe ciphertext
        "signed":$p
    );

    let hasCustomAssemblyFormat = 1;
}

def Concrete_CleartextType : Concrete_Type<"Cleartext"> {
    let mnemonic = "cleartext";

    let summary = "A cleartext (a fixed-precision integer) ready to be multiplied to a LWE ciphertext";

    let description = [{
       Cleartext.
    }];

    let parameters = (ins
        // Number of bits of the cleartext representation
        "signed":$p
    );

    let hasCustomAssemblyFormat = 1;
}

def Concrete_PlaintextType : Concrete_Type<"Plaintext"> {
    let mnemonic = "plaintext";

    let summary = "A Plaintext (a fixed-precision integer) ready to be added to a LWE ciphertext";

    let description = [{
       Plaintext.
    }];

    let parameters = (ins
        // Number of bits of the cleartext representation
        "signed":$p
    );

    let hasCustomAssemblyFormat = 1;
}

def Concrete_Context : Concrete_Type<"Context"> {
    let mnemonic = "context";

    let summary = "A runtime context";

    let description = [{
       An abstract runtime context to pass contextual value, like public keys, ...
    }];
}

#endif
