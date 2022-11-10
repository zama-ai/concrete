//===- TFHEOps.td - High level FHE dialect ops ----------------*- tablegen
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHE_OPS
#define CONCRETELANG_DIALECT_TFHE_IR_TFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/TFHE/IR/TFHEDialect.td"
include "concretelang/Dialect/TFHE/IR/TFHETypes.td"

class TFHE_Op<string mnemonic, list<Trait> traits = []>
    : Op<TFHE_Dialect, mnemonic, traits>;

def TFHE_EncodeExpandLutForBootstrapOp : TFHE_Op<"encode_expand_lut_for_bootstrap"> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a bootstrap.";

    let arguments = (ins
        1DTensorOf<[I64]> : $input_lookup_table,
        I32Attr: $polySize,
        I32Attr: $outputBits
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}

def TFHE_EncodeExpandLutForWopPBSOp : TFHE_Op<"encode_expand_lut_for_woppbs"> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a wop pbs.";

    let arguments = (ins
        1DTensorOf<[I64]> : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $polySize,
        I32Attr : $modulusProduct
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}

def TFHE_EncodePlaintextWithCrtOp : TFHE_Op<"encode_plaintext_with_crt"> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis.";

    let arguments = (ins
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}


def TFHE_ZeroGLWEOp : TFHE_Op<"zero"> {
  let summary = "Returns a trivial encyption of 0";

  let arguments = (ins);
  let results = (outs TFHE_GLWECipherTextType : $out);
}

def TFHE_ZeroTensorGLWEOp : TFHE_Op<"zero_tensor"> {
  let summary = "Returns a tensor of trivial encyption of 0";

  let arguments = (ins);
  let results = (outs Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>:$tensor);
}

def TFHE_AddGLWEIntOp : TFHE_Op<"add_glwe_int"> {
  let summary = "Returns the sum of a clear integer and a lwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a, AnyInteger : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_AddGLWEOp : TFHE_Op<"add_glwe"> {
  let summary = "Returns the sum of 2 lwe ciphertexts";

  let arguments = (ins TFHE_GLWECipherTextType : $a, TFHE_GLWECipherTextType : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_SubGLWEIntOp : TFHE_Op<"sub_int_glwe"> {
  let summary = "Substracts an integer and a GLWE ciphertext";

  let arguments = (ins AnyInteger : $a, TFHE_GLWECipherTextType : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_NegGLWEOp : TFHE_Op<"neg_glwe"> {
  let summary = "Negates a glwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_MulGLWEIntOp : TFHE_Op<"mul_glwe_int"> {
  let summary = "Returns the product of a clear integer and a lwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a, AnyInteger : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_KeySwitchGLWEOp : TFHE_Op<"keyswitch_glwe"> {
  let summary = "Change the encryption parameters of a glwe ciphertext by "
                "applying a keyswitch";

  let arguments = (ins 
    TFHE_GLWECipherTextType : $ciphertext,
    I32Attr : $level,
    I32Attr : $baseLog
  );
  

  let results = (outs TFHE_GLWECipherTextType : $result);
}


def TFHE_BootstrapGLWEOp : TFHE_Op<"bootstrap_glwe"> {
  let summary =
      "Programmable bootstraping of a GLWE ciphertext with a lookup table";

  let arguments = (ins
    TFHE_GLWECipherTextType : $ciphertext,
    1DTensorOf<[I64]> : $lookup_table,
    I32Attr : $level,
    I32Attr : $baseLog,
    I32Attr : $polySize,
    I32Attr : $glweDimension
  );

  let results = (outs TFHE_GLWECipherTextType : $result);
}

def TFHE_WopPBSGLWEOp : TFHE_Op<"wop_pbs_glwe"> {
    let summary = "";

    let arguments = (ins
        Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>: $ciphertexts,
        1DTensorOf<[I64]> : $lookupTable,
        // Bootstrap parameters
        I32Attr : $bootstrapLevel,
        I32Attr : $bootstrapBaseLog,
        // Keyswitch parameters
        I32Attr : $keyswitchLevel,
        I32Attr : $keyswitchBaseLog,
        // Packing keyswitch key parameters
        I32Attr : $packingKeySwitchInputLweDimension,
        I32Attr : $packingKeySwitchoutputPolynomialSize,
        I32Attr : $packingKeySwitchLevel,
        I32Attr : $packingKeySwitchBaseLog,
        // Circuit bootstrap parameters
        I32Attr : $circuitBootstrapLevel,
        I32Attr : $circuitBootstrapBaseLog,
        // Crt decomposition
        I64ArrayAttr: $crtDecomposition
    );
    let results = (outs Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>:$result);
}

#endif
