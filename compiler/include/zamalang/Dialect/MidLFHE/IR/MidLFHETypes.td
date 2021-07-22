#ifndef ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_TYPES
#define ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_TYPES

// TODO: MLWE / GSW

include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.td"
include "mlir/IR/BuiltinTypes.td"

class MidLFHE_Type<string name, list<Trait> traits = []> : TypeDef<MidLFHE_Dialect, name, traits> { }

def GLWECipherTextType : MidLFHE_Type<"GLWECipherText", [MemRefElementTypeInterface]> {
    let mnemonic = "glwe";

    let summary = "An GLWE cipher text";

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

    // We define the printer inline.
    let printer = [{
        $_printer << "glwe"
            << "<{";
                if (getImpl()->dimension == -1) $_printer << "_";
                else $_printer << getImpl()->dimension;
                $_printer << ",";
                if (getImpl()->polynomialSize == -1) $_printer << "_";
                else $_printer << getImpl()->polynomialSize;
                $_printer << ",";
                if (getImpl()->bits == -1) $_printer << "_";
                else $_printer << getImpl()->bits;
        $_printer << "}";
        $_printer << "{";
                if (getImpl()->p == -1) $_printer << "_";
                else $_printer << getImpl()->p;
        $_printer << "}>";
    }];

    // The parser is defined here also.
    let parser = [{
        if ($_parser.parseLess())
            return Type();

        // First parameters block
        if ($_parser.parseLBrace())
            return Type();
        int dimension = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(dimension))
            return Type();
        if ($_parser.parseComma())
            return Type();
        int polynomialSize = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(polynomialSize))
            return Type();
        if ($_parser.parseComma())
            return Type();
        int bits = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(bits))
            return Type();
        if ($_parser.parseRBrace())
            return Type();

        // Next parameters block
        if ($_parser.parseLBrace())
            return Type();
        int p = -1;
        if ($_parser.parseInteger(p))
            return Type();
        if ($_parser.parseRBrace())
            return Type();

        if ($_parser.parseGreater())
            return Type();
        Location loc = $_parser.getEncodedSourceLoc($_parser.getNameLoc());
        return getChecked(loc, loc.getContext(), dimension, polynomialSize, bits, p);
    }];

    let genVerifyDecl = true;
}

#endif
