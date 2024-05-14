// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_RT_TYPEINFERENCE_H
#define CONCRETELANG_DIALECT_RT_TYPEINFERENCE_H

#include <concretelang/Analysis/TypeInferenceAnalysis.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>

namespace mlir {
namespace concretelang {
namespace RT {

template <typename BaseConstraintT, unsigned int leftDepth,
          unsigned int rightDepth>
class SameNestedTypeContraint
    : public SameNestedTypeConstraintBase<BaseConstraintT, leftDepth,
                                          rightDepth> {
public:
  using SameNestedTypeConstraintBase<BaseConstraintT, leftDepth,
                                     rightDepth>::SameNestedTypeConstraintBase;

protected:
  mlir::Type getNestedType(mlir::Type t) override {
    if (concretelang::RT::PointerType ptrt =
            llvm::dyn_cast<concretelang::RT::PointerType>(t)) {
      return ptrt.getElementType();
    } else if (concretelang::RT::FutureType futt =
                   llvm::dyn_cast<concretelang::RT::FutureType>(t)) {
      return futt.getElementType();
    } else {
      return t;
    }
  }

  mlir::Type applyNestedType(mlir::Type nestedType, mlir::Type t) override {
    if (llvm::isa<concretelang::RT::PointerType>(t)) {
      return concretelang::RT::PointerType::get(nestedType);
    } else if (llvm::isa<concretelang::RT::FutureType>(t)) {
      return concretelang::RT::FutureType::get(nestedType);
    } else {
      return t;
    }
  }
};

// Constraint ensuring that two types are identical when the
// indirection through a pointer is stripped. The parameters
// `leftIsPointer` and `rightIsPointer` indicate which types should be
// stripped from the pointer indirection before comparison.
template <typename BaseConstraintT, bool leftIsPointer, bool rightIsPointer>
class SamePointerTypeContraint
    : public SameNestedTypeConstraintBase<
          BaseConstraintT, leftIsPointer ? 1 : 0, rightIsPointer ? 1 : 0> {
public:
  using SameNestedTypeConstraintBase<
      BaseConstraintT, leftIsPointer ? 1 : 0,
      rightIsPointer ? 1 : 0>::SameNestedTypeConstraintBase;

protected:
  mlir::Type getNestedType(mlir::Type t) override {
    concretelang::RT::PointerType ptrt =
        llvm::cast<concretelang::RT::PointerType>(t);
    return ptrt.getElementType();
  }

  mlir::Type applyNestedType(mlir::Type nestedType, mlir::Type t) override {
    if (llvm::isa<concretelang::RT::PointerType>(t)) {
      return concretelang::RT::PointerType::get(nestedType);
    } else {
      return t;
    }
  }
};

// Constraint ensuring that two types are identical when the
// indirection through a future is stripped. The parameters
// `leftIsFuture` and `rightIsFuture` indicate which types should be
// stripped from the future indirection before comparison.
template <typename BaseConstraintT, bool leftIsFuture, bool rightIsFuture>
class SameFutureTypeContraint
    : public SameNestedTypeConstraintBase<BaseConstraintT, leftIsFuture ? 1 : 0,
                                          rightIsFuture ? 1 : 0> {
public:
  using SameNestedTypeConstraintBase<
      BaseConstraintT, leftIsFuture ? 1 : 0,
      rightIsFuture ? 1 : 0>::SameNestedTypeConstraintBase;

protected:
  mlir::Type getNestedType(mlir::Type t) override {
    concretelang::RT::FutureType ptrt =
        llvm::cast<concretelang::RT::FutureType>(t);
    return ptrt.getElementType();
  }

  mlir::Type applyNestedType(mlir::Type nestedType, mlir::Type t) override {
    if (llvm::isa<concretelang::RT::FutureType>(t)) {
      return concretelang::RT::FutureType::get(nestedType);
    } else {
      return t;
    }
  }
};

} // namespace RT
} // namespace concretelang
} // namespace mlir

#endif // CONCRETELANG_DIALECT_RT_TYPEINFERENCE_H
