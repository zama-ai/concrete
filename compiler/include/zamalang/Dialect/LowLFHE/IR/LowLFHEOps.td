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

def BootstrapLweOp : LowLFHE_Op<"bootstrap_lwe"> {
    let arguments = (ins 
        LweBootstrapKeyType:$bootstrap_key,
        LweCiphertextType:$input_ciphertext,
        GlweCiphertextType:$accumulator
    );
    let results = (outs LweCiphertextType:$result);
}

def DecryptGlweOp : LowLFHE_Op<"decrypt_glwe"> {
    let arguments = (ins GlweSecretKeyType:$secret_key, GlweCiphertextType:$ciphertext);
    let results = (outs PlaintextListType:$result);
}

def DecryptLweOp : LowLFHE_Op<"decrypt_lwe"> {
    let arguments = (ins LweSecretKeyType:$secret_key, LweCiphertextType:$ciphertext);
    let results = (outs PlaintextType:$result);
}

def EncryptGlweOp : LowLFHE_Op<"encrypt_glwe"> {
    let arguments = (ins
        GlweSecretKeyType:$secret_key,
        PlaintextListType:$plaintext,
        EncryptionRandomGeneratorType:$generator,
        VarianceType:$variance
    );
    let results = (outs GlweCiphertextType:$result);
}

def EncryptLweOp : LowLFHE_Op<"encrypt_lwe"> {
    let arguments = (ins
        LweSecretKeyType:$secret_key,
        PlaintextType:$plaintext,
        EncryptionRandomGeneratorType:$generator,
        VarianceType:$variance
    );
    let results = (outs LweCiphertextType:$result);
}

def GetPlaintextListElementOp : LowLFHE_Op<"get_plaintext_list_element">{
    let arguments = (ins PlaintextListType:$plaintext, Index:$i);

    let results = (outs AnySignlessInteger:$result);
}

def SetPlaintextListElementOp : LowLFHE_Op<"set_plaintext_list_element">{
    let arguments = (ins PlaintextListType:$plaintext, Index:$i, AnySignlessInteger:$val);

    let results = (outs );
}

def KeySwitchLweOp : LowLFHE_Op<"keyswitch_lwe"> {
    let arguments = (ins LweKeySwitchKeyType:$keyswitch_key, LweCiphertextType:$ciphertext);
    let results = (outs LweCiphertextType:$result);
}


def FillGlweSecretKeyOp : LowLFHE_Op<"fill_glwe_secret_key"> {
    let arguments = (ins GlweSecretKeyType:$secret_key, SecretRandomGeneratorType:$generator);
    let results = (outs );
}

def FillLweSecretKeyOp : LowLFHE_Op<"fill_lwe_secret_key"> {
    let arguments = (ins LweSecretKeyType:$secret_key, SecretRandomGeneratorType:$generator);
    let results = (outs );
}

def FillLweBootstrapKeyOp : LowLFHE_Op<"fill_lwe_bootstrap_key"> {
    let arguments = (ins
        LweBootstrapKeyType:$bootstrap_key,
        LweSecretKeyType:$lwe_secret_key,
        GlweSecretKeyType:$glwe_secret_key,
        EncryptionRandomGeneratorType:$generator,
        VarianceType:$noise
    );
    let results = (outs );
}

def FillLweKeyswitchKeyOp : LowLFHE_Op<"fill_lwe_keyswitch_key"> {
    let arguments = (ins
        LweKeySwitchKeyType:$keyswitch_key,
        LweSecretKeyType:$input_key,
        LweSecretKeyType:$output_key,
        EncryptionRandomGeneratorType:$generator,
        VarianceType:$noise
    );
    let results = (outs );
}

def FillPlaintextListWithExpansionOp : LowLFHE_Op<"fill_plaintext_list_with_expansion"> {
    let arguments = (ins PlaintextListType:$output, ForeignPlaintextListType:$input);
    let results = (outs );
}

def EncodeCleartextOp : LowLFHE_Op<"encode_cleartext"> {
    let arguments = (ins CleartextType:$cleartext);
    let results = (outs PlaintextType:$plaintext);
}

def EncodeIntOp : LowLFHE_Op<"encode_int"> {
    let arguments = (ins AnyInteger:$i);
    let results = (outs PlaintextType:$plaintext);
}

def IntToCleartextOp : LowLFHE_Op<"int_to_cleartext"> {
    let arguments = (ins AnyInteger:$i);
    let results = (outs CleartextType:$cleartext);
}

def ConstCleartextOp : LowLFHE_Op<"const_cleartext", [ConstantLike]> {
    // TODO: maybe restrict to specific attr?
    let arguments = (ins AnyAttr:$value);
    let results = (outs CleartextType:$cleartext);
}

#endif
