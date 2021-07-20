//===- MidLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_OPS
#define ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.td"
include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.td"

class MidLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MidLFHE_Dialect, mnemonic, traits>;

def AddGLWEInt : MidLFHE_Op<"add_glwe_int"> {
    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return mlir::zamalang::MidLFHE::verifyGLWEIntegerOperator<AddGLWEInt>(*this);
    }];
}

def AddGLWEOp : MidLFHE_Op<"add_glwe"> {
    let arguments = (ins GLWECipherTextType:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::zamalang::MidLFHE::verifyBinaryGLWEOperator<AddGLWEOp>(*this);
    }];
}

def SubIntGLWEOp : MidLFHE_Op<"sub_int_glwe"> {
    let arguments = (ins AnyInteger:$a, GLWECipherTextType:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::zamalang::MidLFHE::verifyIntegerGLWEOperator(*this);
    }];
}


def MulGLWEInt : MidLFHE_Op<"mul_glwe_int"> {
    let arguments = (ins GLWECipherTextType:$a, AnyInteger:$b);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return mlir::zamalang::MidLFHE::verifyGLWEIntegerOperator<MulGLWEInt>(*this);
    }];
}



def ApplyLookupTable : MidLFHE_Op<"apply_lookup_table"> {
    let arguments = (ins GLWECipherTextType:$ct,
        MemRefOf<[AnyInteger]>:$l_cst,
        I32Attr:$k, I32Attr:$polynomialSize,
        I32Attr:$levelKS, I32Attr:$baseLogKS,
        I32Attr:$levelBS, I32Attr:$baseLogBS);
    let results = (outs GLWECipherTextType);

    let verifier = [{
        return ::mlir::zamalang::MidLFHE::verifyApplyLookupTable(*this);
    }];
}

#endif
