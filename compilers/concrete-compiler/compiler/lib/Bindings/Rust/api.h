// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang-c/Dialect/FHE.h>
#include <concretelang-c/Dialect/FHELinalg.h>
#include <concretelang-c/Support/CompilerEngine.h>
#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Conversion.h>
#include <mlir-c/Debug.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/Async.h>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/GPU.h>
#include <mlir-c/Dialect/LLVM.h>
#include <mlir-c/Dialect/Linalg.h>
#include <mlir-c/Dialect/PDL.h>
#include <mlir-c/Dialect/Quant.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Shape.h>
#include <mlir-c/Dialect/SparseTensor.h>
#include <mlir-c/Dialect/Tensor.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/IR.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Interfaces.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Registration.h>
#include <mlir-c/Support.h>
#include <mlir-c/Transforms.h>
