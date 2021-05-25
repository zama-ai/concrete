//===- MidLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_OPS
#define ZAMALANG_DIALECT_MidLFHE_IR_MidLFHE_OPS

include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.td"
include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.td"

class MidLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MidLFHE_Dialect, mnemonic, traits>;

#endif
