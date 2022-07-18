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

def TFHE_GLWEFromTableOp : TFHE_Op<"glwe_from_table"> {
  let summary =
      "Creates a GLWE ciphertext which is the trivial encrytion of a the input "
      "table interpreted as a polynomial (to use later in a bootstrap)";

  let arguments = (ins 1DTensorOf < [I64] > : $table);
  let results = (outs TFHE_GLWECipherTextType : $result);
}

def TFHE_BootstrapGLWEOp : TFHE_Op<"bootstrap_glwe"> {
  let summary =
      "Programmable bootstraping of a GLWE ciphertext with a lookup table";

  let arguments = (ins
    TFHE_GLWECipherTextType : $ciphertext,
    TFHE_GLWECipherTextType : $lookup_table,
    I32Attr : $level,
    I32Attr : $baseLog
  );

  let results = (outs TFHE_GLWECipherTextType : $result);
}

#endif
