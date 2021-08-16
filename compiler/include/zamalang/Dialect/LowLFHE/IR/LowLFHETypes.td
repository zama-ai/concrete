#ifndef ZAMALANG_DIALECT_LowLFHE_IR_LowLFHE_TYPES
#define ZAMALANG_DIALECT_LowLFHE_IR_LowLFHE_TYPES

include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.td"

class LowLFHE_Type<string name> : TypeDef<LowLFHE_Dialect, name> { }

def EncryptionRandomGeneratorType : LowLFHE_Type<"EncryptionRandomGenerator"> {
    let mnemonic = "enc_rand_gen";

    let summary = "Random source for encryption";

    let description = [{
       Random source for encryption.
    }];

    let printer = [{
        $_printer << "enc_rand_gen";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def SecretRandomGeneratorType : LowLFHE_Type<"SecretRandomGenerator"> {
    let mnemonic = "secret_rand_gen";

    let summary = "Random source for secret key generation";

    let description = [{
       Random source for secret key generation.
    }];

    let printer = [{
        $_printer << "secret_rand_gen";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}


def PlaintextType : LowLFHE_Type<"Plaintext"> {
    let mnemonic = "plaintext";

    let summary = "Plaintext";

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

def PlaintextListType : LowLFHE_Type<"PlaintextList"> {
    let mnemonic = "plaintext_list";

    let summary = "Plaintext list";

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

def ForeignPlaintextListType : LowLFHE_Type<"ForeignPlaintextList"> {
    let mnemonic = "foreign_plaintext_list";

    let summary = "Foreign plaintext list";

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


def LweCiphertextType : LowLFHE_Type<"LweCiphertext"> {
    let mnemonic = "lwe_ciphertext";

    let summary = "LWE ciphertext";

    let description = [{
       Learning With Error ciphertext.
    }];


    let parameters = (ins
        // The size of the lwe ciphertext
        "signed":$size,
        // Precision of the lwe ciphertext
        "signed":$p
    );

    let printer = [{
        $_printer << "lwe_ciphertext<";
        if (getImpl()->size == -1) $_printer << "_";
        else $_printer << getImpl()->size;
        $_printer << ",";
        if (getImpl()->p == -1) $_printer << "_";
        else $_printer << getImpl()->p;
        $_printer << ">";
    }];

    let parser = [{
        if ($_parser.parseLess())
            return Type();
        int size = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(size))
            return Type();
        if ($_parser.parseComma())
            return Type();
        int p = -1;
        if ($_parser.parseOptionalKeyword("_") && $_parser.parseInteger(p))
            return Type();
        if ($_parser.parseGreater())
            return Type();
        Location loc = $_parser.getEncodedSourceLoc($_parser.getNameLoc());
        return getChecked(loc, loc.getContext(), size, p);
    }];
}

def LweKeySwitchKeyType : LowLFHE_Type<"LweKeySwitchKey"> {
    let mnemonic = "lwe_key_switch_key";

    let summary = "LWE keyswitching key";

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

def LweBootstrapKeyType : LowLFHE_Type<"LweBootstrapKey"> {
    let mnemonic = "lwe_bootstrap_key";

    let summary = "LWE bootstrapping key";

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

def LweSecretKeyType : LowLFHE_Type<"LweSecretKey"> {
    let mnemonic = "lwe_secret_key";

    let summary = "LWE bootstrapping key";

    let description = [{
       Learning With Error bootstrapping key.
    }];

    let printer = [{
        $_printer << "lwe_secret_key";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def LweSizeType : LowLFHE_Type<"LweSize"> {
    let mnemonic = "lwe_size";

    let summary = "LWE size";

    let description = [{
       Learning With Error size.
    }];

    let printer = [{
        $_printer << "lwe_size";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def GlweCiphertextType : LowLFHE_Type<"GlweCiphertext"> {
    let mnemonic = "glwe_ciphertext";

    let summary = "GLWE ciphertext";

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


def GlweSecretKeyType : LowLFHE_Type<"GlweSecretKey"> {
    let mnemonic = "glwe_secret_key";

    let summary = "GLWE secret key";

    let description = [{
       GLWE secret key.
    }];

    let printer = [{
        $_printer << "glwe_secret_key";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def GlweSizeType : LowLFHE_Type<"GlweSize"> {
    let mnemonic = "glwe_size";

    let summary = "GLWE size";

    let description = [{
       GLWE size.
    }];

    let printer = [{
        $_printer << "glwe_size";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def PolynomialSizeType : LowLFHE_Type<"PolynomialSize"> {
    let mnemonic = "polynomial_size";

    let summary = "Polynomial size";

    let description = [{
       Polynomial size.
    }];

    let printer = [{
        $_printer << "polynomial_size";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}


def DecompositionLevelCountType : LowLFHE_Type<"DecompositionLevelCount"> {
    let mnemonic = "decomp_level_count";

    let summary = "Decomposition level";

    let description = [{
       Decomposition level.
    }];

    let printer = [{
        $_printer << "decomp_level_count";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def DecompositionBaseLogType : LowLFHE_Type<"DecompositionBaseLog"> {
    let mnemonic = "decomp_base_log";

    let summary = "Decomposition base log";

    let description = [{
       Decomposition base log.
    }];

    let printer = [{
        $_printer << "decomp_base_log";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}


def VarianceType : LowLFHE_Type<"Variance"> {
    let mnemonic = "variance";

    let summary = "Variance";

    let description = [{
       Variance.
    }];

    let printer = [{
        $_printer << "variance";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}


def CleartextType : LowLFHE_Type<"Cleartext"> {
    let mnemonic = "cleartext";

    let summary = "Cleartext";

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



#endif
