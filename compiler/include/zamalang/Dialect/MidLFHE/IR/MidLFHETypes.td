#ifndef ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_TYPES
#define ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_TYPES

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

#endif
