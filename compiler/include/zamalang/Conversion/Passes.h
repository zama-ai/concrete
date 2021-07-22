//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ZAMALANG_CONVERSION_PASSES_H
#define ZAMALANG_CONVERSION_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "zamalang/Conversion/HLFHEToMidLFHE/Pass.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"

// namespace mlir {
// namespace zamalang {
#define GEN_PASS_CLASSES
#include "zamalang/Conversion/Passes.h.inc"
// } // namespace zamalang
// } // namespace mlir

#endif
