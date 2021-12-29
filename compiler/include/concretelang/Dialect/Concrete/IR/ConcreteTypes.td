#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_TYPES
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_TYPES

include "mlir/IR/BuiltinTypes.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"

class Concrete_Type<string name, list<Trait> traits = []> : TypeDef<Concrete_Dialect, name, traits> { }

def GlweCiphertextType : Concrete_Type<"GlweCiphertext"> {
    let mnemonic = "glwe_ciphertext";

    let summary = "A GLWE ciphertext (encryption of a polynomial of fixed-precision integers)";

    let description = [{
       GLWE ciphertext.
    }];

    let printer = [{
        $_printer << "glwe_ciphertext";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def LweCiphertextType : Concrete_Type<"LweCiphertext", [MemRefElementTypeInterface]> {
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

    let printer = [{
        $_printer << "lwe_ciphertext<";
        if (getImpl()->dimension == -1) $_printer << "_";
        else $_printer << getImpl()->dimension;
        $_printer << ",";
        if (getImpl()->p == -1) $_printer << "_";
        else $_printer << getImpl()->p;
        $_printer << ">";
    }];

    let parser = [{
        if ($_parser.parseLess())
            return Type();
        int dimension = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(dimension))
            return Type();
        if ($_parser.parseComma())
            return Type();
        int p = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(p))
            return Type();
        if ($_parser.parseGreater())
            return Type();
        Location loc = $_parser.getEncodedSourceLoc($_parser.getNameLoc());
        return getChecked(loc, loc.getContext(), dimension, p);
    }];
}

def CleartextType : Concrete_Type<"Cleartext"> {
    let mnemonic = "cleartext";

    let summary = "A cleartext (a fixed-precision integer) ready to be multiplied to a LWE ciphertext";

    let description = [{
       Cleartext.
    }];

    let parameters = (ins
        // Number of bits of the cleartext representation
        "signed":$p
    );

    let printer = [{
        $_printer << "cleartext<";
        if (getImpl()->p == -1) $_printer << "_";
        else $_printer << getImpl()->p;
        $_printer << ">";
    }];

    let parser = [{
        if ($_parser.parseLess())
            return Type();
        int p = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(p))
            return Type();
        if ($_parser.parseGreater())
            return Type();
        Location loc = $_parser.getEncodedSourceLoc($_parser.getNameLoc());
        return getChecked(loc, loc.getContext(), p);
    }];
}

def PlaintextType : Concrete_Type<"Plaintext"> {
    let mnemonic = "plaintext";

    let summary = "A Plaintext (a fixed-precision integer) ready to be added to a LWE ciphertext";

    let description = [{
       Plaintext.
    }];

    let parameters = (ins
        // Number of bits of the cleartext representation
        "signed":$p
    );

    let printer = [{
        $_printer << "plaintext<";
        if (getImpl()->p == -1) $_printer << "_";
        else $_printer << getImpl()->p;
        $_printer << ">";
    }];

    let parser = [{
        if ($_parser.parseLess())
            return Type();
        int p = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(p))
            return Type();
        if ($_parser.parseGreater())
            return Type();
        Location loc = $_parser.getEncodedSourceLoc($_parser.getNameLoc());
        return getChecked(loc, loc.getContext(), p);
    }];
}

def PlaintextListType : Concrete_Type<"PlaintextList"> {
    let mnemonic = "plaintext_list";

    let summary = "List of plaintexts";

    let description = [{
       Plaintext list.
    }];

    let printer = [{
        $_printer << "plaintext_list";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def ForeignPlaintextListType : Concrete_Type<"ForeignPlaintextList"> {
    let mnemonic = "foreign_plaintext_list";

    let summary = "A foreign (reference to a independently allocated memory space) plaintext list";

    let description = [{
       Foreign plaintext list.
    }];

    let printer = [{
        $_printer << "foreign_plaintext_list";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def LweKeySwitchKeyType : Concrete_Type<"LweKeySwitchKey"> {
    let mnemonic = "lwe_key_switch_key";

    let summary = "A LWE keyswitching key";

    let description = [{
       Learning With Error keyswitching key.
    }];

    let printer = [{
        $_printer << "lwe_key_switch_key";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def LweBootstrapKeyType : Concrete_Type<"LweBootstrapKey"> {
    let mnemonic = "lwe_bootstrap_key";

    let summary = "A LWE bootstrapping key";

    let description = [{
       Learning With Error bootstrapping key.
    }];

    let printer = [{
        $_printer << "lwe_bootstrap_key";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def Context : Concrete_Type<"Context"> {
    let mnemonic = "context";

    let summary = "A runtime context";

    let description = [{
       An abstract runtime context to pass contextual value, like public keys, ...
    }];

    let printer = [{
        $_printer << "context";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}



#endif
