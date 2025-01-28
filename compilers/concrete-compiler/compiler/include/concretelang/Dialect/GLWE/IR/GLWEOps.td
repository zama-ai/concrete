//===- GLWEOps.td - High level GLWE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_OPS
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


include "concretelang/Dialect/GLWE/IR/GLWEDialect.td"
include "concretelang/Dialect/GLWE/IR/GLWETypes.td"
include "concretelang/Dialect/GLWE/IR/GLWEAttrs.td"

// include "concretelang/Dialect/GLWE/Interfaces/GLWEInterfaces.td"

// class GLWEParameterMatchPred<
//   string v1, string v2, list<string> parameters,
//   string comparator = "std::equal_to<>()"
// > :
// PredOpTrait<
//     "the " # v1 # " all of GLWE {" # !interleave(parameters, ", ") # "} parameters matches the " # v2 # " value",
//     And<!foreach(parameter, parameters, CPred<
//           comparator # "(" #
//           "::llvm::cast<GLWEType>($" # v1 # ".getType()).getParams().get" # parameter # "(), " #
//           "::llvm::cast<GLWEType>($" # v2 # ".getType()).getParams().get" # parameter # "())"
//         >
//     )>
// >; 


class GLWE_Op<string mnemonic, list<Trait> traits = []> :
    Op<GLWE_Dialect, mnemonic, traits>;

def GLWE_ModulusSwitching: GLWE_Op<"modulus_switching"
    //,[GLWEParameterMatchPred<"input", "output", ["MessageBound", "MaskModulus"]>]
> {
    let arguments = (ins
        GLWE_GLWE:$input,
        I64Attr: $modulus,
        DefaultValuedAttr<BoolAttr, "false">: $partial
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";

    let hasVerifier = 1;
}

def GLWE_ExactDecomposition: GLWE_Op<"exact_decomposition"> {
    let arguments = (ins
        GLWE_GLWE:$input,
        GLWE_DecompositionParameters:$decomp,
        DefaultValuedAttr<BoolAttr, "false">: $partial
    );

    let results = (outs GLWE_RadixGLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def GLWE_ExactRecomposition: GLWE_Op<"exact_recomposition"> {
    let arguments = (ins
        GLWE_RadixGLWE:$input,
        GLWE_GLev:$glev,
        DefaultValuedAttr<BoolAttr, "false">: $fft // TODO?
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` `(` type($input) `,` type($glev) `)` `->` type($output)";
}

def GLWE_SampleExtract: GLWE_Op<"sample_extract"> {
    let arguments = (ins
        GLWE_GLWE:$input
    );

    let results = (outs GLWE_GLWE:$output);

    let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

#endif
