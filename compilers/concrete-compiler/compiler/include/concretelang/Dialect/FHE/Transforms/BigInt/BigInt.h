// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_FHE_BIGINT_PASS_H
#define CONCRETELANG_FHE_BIGINT_PASS_H

#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/FHE/Transforms/BigInt/BigInt.h.inc>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<>>
createFHEBigIntTransformPass(unsigned int chunkSize, unsigned int chunkWidth);

} // namespace concretelang
} // namespace mlir

#endif
