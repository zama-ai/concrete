//===- MidLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_MidLFHE_IR_MidLFHE_OPS
#define CONCRETELANG_DIALECT_MidLFHE_IR_MidLFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/MidLFHE/IR/MidLFHEDialect.td"
include "concretelang/Dialect/MidLFHE/IR/MidLFHETypes.td"

class MidLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MidLFHE_Dialect, mnemonic, traits>;

def ZeroGLWEOp : MidLFHE_Op<"zero"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs GLWECipherTextType:$out);
}

def AddGLWEIntOp : MidLFHE_Op<"add_glwe_int"> {
    let summary = "Returns the sum of a clear integer and a lwe ciphertext";    

    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return mlir::concretelang::MidLFHE::verifyGLWEIntegerOperator<AddGLWEIntOp>(*this);
    }];
}

def AddGLWEOp : MidLFHE_Op<"add_glwe"> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins GLWECipherTextType:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::MidLFHE::verifyBinaryGLWEOperator<AddGLWEOp>(*this);
    }];
}

def SubIntGLWEOp : MidLFHE_Op<"sub_int_glwe"> {
    let summary = "Substracts an integer and a GLWE ciphertext";

    let arguments = (ins AnyInteger:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::MidLFHE::verifyIntegerGLWEOperator(*this);
    }];
}

def NegGLWEOp : MidLFHE_Op<"neg_glwe"> {
    let summary = "Negates a glwe ciphertext";

    let arguments = (ins GLWECipherTextType:$a);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::MidLFHE::verifyUnaryGLWEOperator<NegGLWEOp>(*this);
    }];
}


def MulGLWEIntOp : MidLFHE_Op<"mul_glwe_int"> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return mlir::concretelang::MidLFHE::verifyGLWEIntegerOperator<MulGLWEIntOp>(*this);
    }];
}



def ApplyLookupTable : MidLFHE_Op<"apply_lookup_table"> {
    let summary = "Applies a lookup table to a GLWE ciphertext";

    
    let arguments = (ins GLWECipherTextType:$ct,
        TensorOf<[AnyInteger]>:$l_cst,
        I32Attr:$glweDimension, I32Attr:$polynomialSize,
        I32Attr:$levelKS, I32Attr:$baseLogKS,
        I32Attr:$levelBS, I32Attr:$baseLogBS,
        I32Attr:$outputSizeKS);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::MidLFHE::verifyApplyLookupTable(*this);
    }];
}

#endif
