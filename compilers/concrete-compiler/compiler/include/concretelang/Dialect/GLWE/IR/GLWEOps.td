//===- GLWEOps.td - High level GLWE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWE_OPS
#define CONCRETELANG_DIALECT_GLWE_IR_GLWE_OPS

include "concretelang/Dialect/GLWE/IR/GLWEDialect.td"
include "concretelang/Dialect/GLWE/IR/GLWETypes.td"
include "concretelang/Dialect/GLWE/IR/GLWEAttrs.td"

class GLWE_Op<string mnemonic, list<Trait> traits = []> :
    Op<GLWE_Dialect, mnemonic, traits>;

#endif
