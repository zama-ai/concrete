// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_TRANSFORMS_PASSES_H
#define ZAMALANG_TRANSFORMS_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "zamalang/Conversion/HLFHETensorOpsToLinalg/Pass.h"
#include "zamalang/Conversion/HLFHEToMidLFHE/Pass.h"
#include "zamalang/Conversion/LowLFHEToConcreteCAPI/Pass.h"
#include "zamalang/Conversion/LowLFHEUnparametrize/Pass.h"
#include "zamalang/Conversion/MLIRLowerableDialectsToLLVM/Pass.h"
#include "zamalang/Conversion/MidLFHEGlobalParametrization/Pass.h"
#include "zamalang/Conversion/MidLFHEToLowLFHE/Pass.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"

#define GEN_PASS_CLASSES
#include "zamalang/Conversion/Passes.h.inc"

#endif
