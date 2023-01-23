// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include <concretelang/Dialect/FHE/Analysis/utils.h>

namespace mlir {
namespace concretelang {
namespace fhe {
namespace utils {
/// Returns `true` if the given value is a scalar or tensor argument of
/// a function, for which a MANP of 1 can be assumed.
bool isEncryptedValue(mlir::Value value) {
  return (
      value.getType().isa<mlir::concretelang::FHE::EncryptedIntegerType>() ||
      value.getType().isa<mlir::concretelang::FHE::EncryptedBooleanType>() ||
      (value.getType().isa<mlir::TensorType>() &&
       value.getType()
           .cast<mlir::TensorType>()
           .getElementType()
           .isa<mlir::concretelang::FHE::EncryptedIntegerType>()));
}

/// Returns the bit width of `value` if `value` is an encrypted integer,
/// or the number of bits to represent a boolean if `value` is an encrypted
/// boolean, or the bit width of the elements if `value` is a tensor of
/// encrypted integers.
unsigned int getEintPrecision(mlir::Value value) {
  if (auto ty = value.getType()
                    .dyn_cast_or_null<
                        mlir::concretelang::FHE::EncryptedIntegerType>()) {
    return ty.getWidth();
  }
  if (auto ty = value.getType()
                    .dyn_cast_or_null<
                        mlir::concretelang::FHE::EncryptedBooleanType>()) {
    return mlir::concretelang::FHE::EncryptedBooleanType::getWidth();
  } else if (auto tensorTy =
                 value.getType().dyn_cast_or_null<mlir::TensorType>()) {
    if (auto ty = tensorTy.getElementType()
                      .dyn_cast_or_null<
                          mlir::concretelang::FHE::EncryptedIntegerType>())
      return ty.getWidth();
  }

  assert(false &&
         "Value is neither an encrypted integer nor a tensor of encrypted "
         "integers");

  return 0;
}

} // namespace utils
} // namespace fhe
} // namespace concretelang
} // namespace mlir
