//===- TFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
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

class TFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFHE_Dialect, mnemonic, traits>;

def ZeroGLWEOp : TFHE_Op<"zero"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs GLWECipherTextType:$out);
}

def AddGLWEIntOp : TFHE_Op<"add_glwe_int"> {
    let summary = "Returns the sum of a clear integer and a lwe ciphertext";    

    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return mlir::concretelang::TFHE::verifyGLWEIntegerOperator<AddGLWEIntOp>(*this);
    }];
}

def AddGLWEOp : TFHE_Op<"add_glwe"> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins GLWECipherTextType:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::TFHE::verifyBinaryGLWEOperator<AddGLWEOp>(*this);
    }];
}

def SubIntGLWEOp : TFHE_Op<"sub_int_glwe"> {
    let summary = "Substracts an integer and a GLWE ciphertext";

    let arguments = (ins AnyInteger:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::TFHE::verifyIntegerGLWEOperator(*this);
    }];
}

def NegGLWEOp : TFHE_Op<"neg_glwe"> {
    let summary = "Negates a glwe ciphertext";

    let arguments = (ins GLWECipherTextType:$a);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::TFHE::verifyUnaryGLWEOperator<NegGLWEOp>(*this);
    }];
}


def MulGLWEIntOp : TFHE_Op<"mul_glwe_int"> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return mlir::concretelang::TFHE::verifyGLWEIntegerOperator<MulGLWEIntOp>(*this);
    }];
}



def ApplyLookupTable : TFHE_Op<"apply_lookup_table"> {
    let summary = "Applies a lookup table to a GLWE ciphertext";

    
    let arguments = (ins GLWECipherTextType:$ct,
        TensorOf<[AnyInteger]>:$l_cst,
        I32Attr:$glweDimension, I32Attr:$polynomialSize,
        I32Attr:$levelKS, I32Attr:$baseLogKS,
        I32Attr:$levelBS, I32Attr:$baseLogBS,
        I32Attr:$outputSizeKS);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::concretelang::TFHE::verifyApplyLookupTable(*this);
    }];
}

#endif
