//===- HLFHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHE_OPS
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHE_OPS

include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.td"

class HLFHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<HLFHE_Dialect, mnemonic, traits>;

#endif
