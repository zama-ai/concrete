#ifndef ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_TYPES
#define ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_TYPES

// TODO: MLWE / GSW

include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.td"

class MidLFHE_Type<string name> : TypeDef<MidLFHE_Dialect, name> { }

def LWECipherTextType : MidLFHE_Type<"LWECipherText"> {
    let mnemonic = "lwe";

    let summary = "An LWE cipher text";

    let description = [{
        An LWE cipher text
    }];

    let parameters = (ins "unsigned":$size);

    // We define the printer inline.
    let printer = [{
        $_printer << "lwe<" << getImpl()->size << ">";
    }];

    // The parser is defined here also.
    let parser = [{
    if ($_parser.parseLess())
        return Type();
    int size;
    if ($_parser.parseInteger(size))
        return Type();
    if ($_parser.parseGreater())
        return Type();
    return get($_ctxt, size);
    }];
}

def GLWECipherTextType : MidLFHE_Type<"GLWECipherText"> {
    let mnemonic = "glwe";

    let summary = "An GLWE cipher text";

    let description = [{
        An GLWE cipher text
    }];

    let parameters = (ins "unsigned":$size, "unsigned":$N);

    // We define the printer inline.
    let printer = [{
        $_printer << "glwe<" << getImpl()->size << "," << getImpl()->N << ">";
    }];

    // The parser is defined here also.
    let parser = [{
    if ($_parser.parseLess())
        return Type();
    int size;
    if ($_parser.parseInteger(size))
        return Type();
    if ($_parser.parseComma())
        return Type();
    int N;
    if ($_parser.parseInteger(N))
        return Type();
    if ($_parser.parseGreater())
        return Type();
    return get($_ctxt, size, N);
    }];
}

def GGSWCipherTextType : MidLFHE_Type<"GGSWCipherText"> {
    let mnemonic = "ggsw";

    let summary = "An GGSW cipher text";

    let description = [{
        An GGSW cipher text
    }];

    let parameters = (ins "unsigned":$size, "unsigned":$N, "unsigned": $level, "unsigned":$base_log);

    // We define the printer inline.
    let printer = [{
        $_printer << "ggsw<" << getImpl()->size << "," << getImpl()->N << "," << getImpl()->level << "," << getImpl()->base_log << ">";
    }];

    // The parser is defined here also.
    let parser = [{
    if ($_parser.parseLess())
        return Type();
    int size;
    if ($_parser.parseInteger(size))
        return Type();
    if ($_parser.parseComma())
        return Type();
    int N;
    if ($_parser.parseInteger(N))
        return Type();
    if ($_parser.parseComma())
        return Type();
    int level;
    if ($_parser.parseInteger(level))
        return Type();
    if ($_parser.parseComma())
        return Type();
    int base_log;
    if ($_parser.parseInteger(base_log))
        return Type();
    if ($_parser.parseGreater())
        return Type();
    return get($_ctxt, size, N, level, base_log);
    }];
}

def AnyCipherTextType : MidLFHE_Type<"AnyCipherText"> {
    let mnemonic = "ciphertext";

    let summary = "Any cipher text";

    let description = [{
        Any cipher text
    }];

    // We define the printer inline.
    let printer = [{
        $_printer << "ciphertext";
    }];

    // The parser is defined here also.
    let parser = [{
        return get($_ctxt);
    }];
}

def CipherTextType: TypeConstraint<Or<[
    LWECipherTextType.predicate,
    GLWECipherTextType.predicate,
    GGSWCipherTextType.predicate,
    AnyCipherTextType.predicate,
    ]>, "ciphertext-like">;

#endif
