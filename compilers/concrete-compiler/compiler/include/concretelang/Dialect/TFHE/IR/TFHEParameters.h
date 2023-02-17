// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHEPARAMETERS_H
#define CONCRETELANG_DIALECT_TFHE_IR_TFHEPARAMETERS_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

namespace mlir {
namespace concretelang {
namespace TFHE {

/// A type parameter representing GLWE secret key.
///
/// A glwe secret key is basically a glwe dimension, a polynomial size, and an
/// id that makes it possible to disambiguate potential keys with with same
/// parameters.
///
/// Note that a key can be instantiated to a `none` key, to serve as a
/// placeholder in the IR. In this case, none of its data are actually usable
/// for lowering to the `Concrete` dialect. Once the
/// `TFHEGlobalParameterization` was performed, there should remain no such
/// `none` keys in the IR.
class GLWESecretKey {
public:
  /// Creates a new none key.
  GLWESecretKey();
  /// Create a new key from parameters.
  GLWESecretKey(int64_t dimension, int64_t polySize, int64_t id);
  bool operator==(GLWESecretKey other);
  bool operator==(const GLWESecretKey other) const;
  bool operator!=(GLWESecretKey other);
  /// Returns the dimension associated with this key, if the key is not none.
  mlir::Optional<int64_t> getDimension() const;
  /// Returns the polynomial size associated with this key, if the key is not
  /// none.
  mlir::Optional<int64_t> getPolySize() const;
  /// Returns the id associated with this key, if the key is not none.
  mlir::Optional<int64_t> getId() const;
  /// Returns true if the key was not filled with valid parameters.
  bool isNotParameterized() const;

private:
  int64_t dimension;
  int64_t polySize;
  int64_t id;
};

llvm::hash_code hash_value(const GLWESecretKey &key);
} // namespace TFHE
} // namespace concretelang
} // namespace mlir

namespace mlir {

AsmPrinter &operator<<(AsmPrinter &p,
                       mlir::concretelang::TFHE::GLWESecretKey key);

template <> struct FieldParser<mlir::concretelang::TFHE::GLWESecretKey> {
  static FailureOr<mlir::concretelang::TFHE::GLWESecretKey>
  parse(AsmParser &parser);
};
} // namespace mlir

#endif
