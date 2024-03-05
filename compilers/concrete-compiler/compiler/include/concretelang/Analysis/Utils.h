// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_ANALYSIS_UTILS_H
#define CONCRETELANG_ANALYSIS_UTILS_H

#include <boost/outcome.h>
#include <concretelang/Common/Error.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Location.h>

namespace mlir {
namespace concretelang {

/// Get the string representation of a location
std::string locationString(mlir::Location loc);

/// Compute the number of iterations based on loop info
int64_t calculateNumberOfIterations(int64_t start, int64_t stop, int64_t step);

/// Compute the number of iterations of an scf for loop
outcome::checked<int64_t, ::concretelang::error::StringError>
calculateNumberOfIterations(scf::ForOp &op);

} // namespace concretelang
} // namespace mlir

#endif
