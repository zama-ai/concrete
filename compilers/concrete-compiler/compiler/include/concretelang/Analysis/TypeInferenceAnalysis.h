// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

// Infrastructure for type inference
//
// The type inference framework below allows for the implementation of
// custom type inference passes that propagate the types of a
// partially-typed input IR throughout its operations following the
// flow of data.
//
// The framework relies on three main concepts:
//
// - A type resolver inheriting from `TypeResolver` that (1) specifies
//   which types are to be considered as unresolved and that (2)
//   resolves the actual types for the values related to an operation
//   based on a previous state of type inference.
//
// - The propagation of types is implemented through the two classes
//   `ForwardTypeInferenceAnalysis` and
//   `BackwardTypeInferenceAnalysis`, which can be used as forward and
//   backward dataflow analyses with the MLIR sparse dataflow analysis
//   framework. These analyses invoke the above-mentioned type
//   resolver for an operation every time the inferred type of a value
//   referenced by that operation changes.
//
// - The type inference state for an operation is represented by an
//   instance of the class `LocalInferenceState`, which maps the
//   values related to an operation to instances of `InferredType`
//   (either indicating the inferred type as an `mlir::Type` or
//   indicating that no type has been inferred, yet).
//
// Additionally, the local rules specifying the relationship between
// the state of inference before invocation of the type resolver and
// the state of inference the invocation can be implemented with type
// constraints (instances of sub-classes of `TypeConstraint`). Type
// constraints can be combined into a `TypeConstraintSet`, which
// provides a method `converge()`, that attempts to apply the
// constraints until the resulting type inference state converges.
//
// There are multiple, predefined type constraint classes for common
// constraints (e.g., if two values must have the same type or the
// same element type). These exist both as static constraints and as
// dynamic constraints. Some pre-defined type constraints depend on a
// class that yields a pair of values for which the contraints shall
// be applied (e.g., yielding two operands or an operand and a result,
// etc.).
//
// The global state of type inference after running the type inference
// analyses is contained in the `DataFlowSolver` to which the analyses
// where added. The local state of inference for an operation can be
// obtained at any stage of the anlysis via the helper methods of
// `TypeInferenceUtils`.

#ifndef CONCRETELANG_ANALYSIS_TYPEINFERENCEANALYSIS_H
#define CONCRETELANG_ANALYSIS_TYPEINFERENCEANALYSIS_H

#include <concretelang/Dialect/TypeInference/IR/TypeInferenceOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>

namespace mlir {
namespace concretelang {
// Encapsulated the state of inference for a single type
class InferredType {
public:
  InferredType() : type(std::nullopt) {}
  InferredType(std::optional<mlir::Type> t) : type(t) {}
  InferredType(mlir::Type t) : type(t) {}

  mlir::Type getType() const { return type.value(); }
  void setType(std::optional<mlir::Type> t) { type = t; }
  bool hasType() const { return type.has_value(); }

  bool operator==(const InferredType &other) const {
    return other.type == type;
  }
  bool operator!=(const InferredType &other) const {
    return other.type != type;
  }

protected:
  std::optional<mlir::Type> type;
};

inline raw_ostream &operator<<(llvm::raw_ostream &os, InferredType t) {
  if (t.hasType())
    os << t.getType();
  else
    os << "(none)";

  return os;
}

// A local inference state is a mapping of values to inferred types
// used to represent the state of type inference for all values
// related to an operation (e.g., types inferred for operands,
// results, block arguments, etc.)
class LocalInferenceState {
public:
  using MapT = llvm::DenseMap<mlir::Value, InferredType>;

  LocalInferenceState() : changed(false) {}

  // Updates the inferred type for a value
  void set(mlir::Value v, InferredType t) {
    if (find(v) != t)
      changed = true;

    state[v] = t;
  }

  // Looks up the inferred type as an `InferredType` for a value. If
  // no type has been inferred or set for the value, an empty inferred
  // type is returned.
  InferredType find(mlir::Value v) const {
    auto it = state.find(v);

    return (it != state.end()) ? it->second : InferredType(std::nullopt);
  }

  // Looks up the inferred type as an MLIR type for a value. If no
  // type has been inferred or set for the value, an empty type is
  // returned.
  mlir::Type lookup(mlir::Value v) const {
    InferredType t = this->find(v);

    return t.hasType() ? t.getType() : mlir::Type();
  }

  // Looks up the inferred types for an entire range of values. For
  // values whose type has not been inferred, empty types are
  // returned.
  llvm::SmallVector<mlir::Type> lookup(mlir::ValueRange r) const {
    return llvm::to_vector(
        llvm::map_range(r, [&](mlir::Value v) { return this->lookup(v); }));
  }

  // Resets the change flag
  void setUnchanged() { changed = false; }

  // Indicates if the inferred type for any of the values has changed
  // or if the inferred type for a new value has been added since the
  // creation of the inference set or since the last time the change
  // flag was reset.
  bool hasChanged() { return changed; }

  // Helper class representing the values for which types have been
  // inferred
  class ValueSet {
  public:
    ValueSet(const MapT &map) : map(map) {}

    template <typename MapIteratorT> class IteratorImpl {
    public:
      IteratorImpl(MapIteratorT it) : it(it) {}
      mlir::Value operator*() { return it->first; }
      mlir::Value *operator->() { return &it->first; }
      IteratorImpl operator++() { return IteratorImpl(it++); }
      bool operator==(const IteratorImpl &other) { return other.it == it; }
      bool operator!=(const IteratorImpl &other) { return other.it != it; }

    protected:
      MapIteratorT it;
    };

    using const_iterator = IteratorImpl<MapT::const_iterator>;

    const_iterator cbegin() const { return const_iterator(map.begin()); }
    const_iterator cend() const { return const_iterator(map.end()); }
    const_iterator begin() const { return const_iterator(map.begin()); }
    const_iterator end() const { return const_iterator(map.end()); }

  protected:
    const MapT &map;
  };

  void dump() const {
    llvm::dbgs() << "LocalInferenceState:\n";

    for (auto [v, t] : state) {
      llvm::dbgs() << "  " << v << ": " << t << "\n";
    }
  }

  const ValueSet getValues() const { return ValueSet(state); }

protected:
  MapT state;
  bool changed;
};

// Lattice value for type inference analysis, encapsulating a single
// inferred type
class TypeInferenceLatticeValue {
public:
  TypeInferenceLatticeValue(const InferredType &t = {}) : inferredType(t) {}

  static TypeInferenceLatticeValue getPessimisticValueState(mlir::Value value) {
    return TypeInferenceLatticeValue();
  }

  bool operator==(const TypeInferenceLatticeValue &rhs) {
    return rhs.getInferredType() == inferredType;
  }

  static TypeInferenceLatticeValue join(const TypeInferenceLatticeValue &lhs,
                                        const TypeInferenceLatticeValue &rhs) {
    return TypeInferenceLatticeValue(lhs.getInferredType().hasType()
                                         ? lhs.getInferredType()
                                         : rhs.getInferredType());
  }

  static TypeInferenceLatticeValue meet(const TypeInferenceLatticeValue &lhs,
                                        const TypeInferenceLatticeValue &rhs) {
    return join(lhs, rhs);
  }

  const InferredType &getInferredType() const { return inferredType; }
  void setType(const InferredType &t) { this->inferredType = t; }

  void print(raw_ostream &os) const { os << inferredType; }

protected:
  InferredType inferredType;
};

// Lattice for forward and backward type inference analysis
class TypeInferenceLattice
    : public mlir::dataflow::Lattice<TypeInferenceLatticeValue> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeInferenceLattice)

  using Lattice::Lattice;
};

// Collection of common operations for forward and backward type
// inference and subsequent transformations (e.g., rewriting the IR
// with the inferred types).
class TypeInferenceUtils {
public:
  // Invokes the callback function `fn` to all values related to the
  // operation `op`. Iteration over the related value stops if `fn`
  // returns `false`.
  //
  // Returns `true` if `fn` returned `true` for all related values,
  // otherwise `false`.
  static bool iterateRelatedValues(mlir::Operation *op,
                                   llvm::function_ref<bool(mlir::Value)> fn) {
    auto applyToOperandsAndResults = [&](mlir::Operation *op) {
      for (mlir::Value v : op->getOperands()) {
        if (!fn(v))
          return false;
      }

      for (mlir::Value v : op->getResults()) {
        if (!fn(v))
          return false;
      }

      return true;
    };

    if (!applyToOperandsAndResults(op))
      return false;

    for (mlir::Region &r : op->getRegions()) {
      for (mlir::Value v : r.getArguments()) {
        if (!fn(v))
          return false;
      }

      if (!op->hasTrait<mlir::OpTrait::NoTerminator>()) {
        // Also map return-like terminators, as they are excluded from
        // forward and backward analysis and not visited individually
        for (mlir::Block &b : r.getBlocks()) {
          mlir::Operation *terminator = b.getTerminator();

          if (terminator->hasTrait<OpTrait::ReturnLike>())
            if (!applyToOperandsAndResults(terminator))
              return false;
        }
      }
    }

    return true;
  }

  // Looks up the inferred types for all values related to the
  // operation `op`, i.e., the inferred types for operands, results,
  // region arguments and operands of terminators used in directly
  // nested regions.
  static LocalInferenceState getLocalInferenceState(
      mlir::Operation *op,
      llvm::function_ref<const TypeInferenceLattice *(mlir::Value)> lookup) {
    LocalInferenceState map;

    iterateRelatedValues(op, [&](mlir::Value v) {
      map.set(v, getInferredType(v, lookup));
      return true;
    });

    return map;
  }

  // Retrieves the inferred types for all values related to `op` using
  // `DataFlowSolver::lookupState` invoked on `solver`.
  static LocalInferenceState
  getLocalInferenceState(const DataFlowSolver &solver, mlir::Operation *op) {
    LocalInferenceState map =
        TypeInferenceUtils::getLocalInferenceState(op, [&](mlir::Value v) {
          return solver.lookupState<TypeInferenceLattice>(v);
        });

    return map;
  }

  // If `t` is a tensor or memref type, return a tensor or memref type
  // with the same shape, but with `elementType` as the element
  // type. If `t` is a scalar type, simply return `elementType`.
  static mlir::Type applyElementType(mlir::Type elementType, mlir::Type t) {
    if (mlir::RankedTensorType rtt =
            llvm::dyn_cast<mlir::RankedTensorType>(t)) {
      return mlir::RankedTensorType::get(
          rtt.getShape(), applyElementType(elementType, rtt.getElementType()));
    } else if (mlir::MemRefType mrt = llvm::dyn_cast<mlir::MemRefType>(t)) {
      return mlir::MemRefType::get(
          mrt.getShape(), applyElementType(elementType, mrt.getElementType()));
    } else {
      return elementType;
    }
  }

protected:
  // Looks up the lattice value for `v` using `lookup` and returns the
  // result as an instance of `InferredType`
  static InferredType getInferredType(
      mlir::Value v,
      llvm::function_ref<const TypeInferenceLattice *(mlir::Value)> lookup) {
    const TypeInferenceLattice *latticeOperand = lookup(v);

    return InferredType(latticeOperand->getValue().getInferredType());
  }
};

// A type resolver is responsible for the actual inference of the
// types related to an operation based on the previous inference state
class TypeResolver {
public:
  virtual ~TypeResolver() {}

  // Resolve the types for all values related to `op` based on the
  // previous inference state `prevState` and the current state
  // `currState`.
  virtual LocalInferenceState resolve(mlir::Operation *op,
                                      const LocalInferenceState &prevState) = 0;

  // Returns true for any type that is considered unresolved by the
  // resolver
  virtual bool isUnresolvedType(mlir::Type t) const = 0;
};

// Base class for all type constraints that can be added to a
// `TypeConstraintSet`.
class TypeConstraint {
public:
  virtual ~TypeConstraint() {}

  // Apply the constraint on the types inferred previously (based on
  // types inferred during previous visits of related operations or
  // the operation itself) and those inferred during the convergence
  // of the current visit of the operation.
  virtual void apply(mlir::Operation *op, TypeResolver &resolver,
                     LocalInferenceState &nextState,
                     const LocalInferenceState &prevState) = 0;

protected:
  // Returns the type inferred for `v`. This is either the type of `v`
  // if `v` has a non-unresolved type, or the type specified in
  // `currState`, or the type specified in `prevState`, or
  // `std::nullopt` if the type of `v` is unresolved and no type has
  // been inferred.
  static InferredType getFirstTypeInOrder(TypeResolver &resolver,
                                          const LocalInferenceState &currState,
                                          const LocalInferenceState &prevState,
                                          mlir::Value v) {
    if (!resolver.isUnresolvedType(v.getType()))
      return v.getType();

    InferredType sit = currState.find(v);

    if (sit.hasType())
      return sit;

    return prevState.find(v);
  }
};

// Yields the operands with the static indices i and j of an operation
template <int i, int j> class OperandValueYield {
public:
  static std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {op->getOperand(i), op->getOperand(j)};
  }
};

// Yields the results with the static indices i and j of an
// operation
template <int i, int j> class ResultValueYield {
public:
  static std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {op->getResult(i), op->getResult(j)};
  }
};

// Yields the operand and result with the static indices i and j of an
// operation
template <int i, int j> class OperandAndResultValueYield {
public:
  static std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {op->getOperand(i), op->getResult(j)};
  }
};

// Yields the i-th and j-th operands of an operation, where `i` and
// `j` are defined dynamically
class DynamicOperandValueYield {
public:
  DynamicOperandValueYield(int i, int j) : i(i), j(j) {}

  std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {op->getOperand(i), op->getOperand(j)};
  }

protected:
  int i;
  int j;
};

// Yields the i-th and j-th results of an operation, where `i` and
// `j` are defined dynamically
class DynamicResultValueYield {
public:
  DynamicResultValueYield(int i, int j) : i(i), j(j) {}

  std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {op->getResult(i), op->getResult(j)};
  }

protected:
  int i;
  int j;
};

// Yields the i-th operand and j-th result of an operation, where `i`
// and `j` are defined dynamically
class DynamicOperandAndResultValueYield {
public:
  DynamicOperandAndResultValueYield(int i, int j) : i(i), j(j) {}

  std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {op->getOperand(i), op->getResult(j)};
  }

protected:
  int i;
  int j;
};

// Yield the return values of two callback functions `f̀` and `g`
class DynamicFunctorYield {
public:
  DynamicFunctorYield(std::function<mlir::Value(void)> f,
                      std::function<mlir::Value(void)> g)
      : f(f), g(g) {}

  std::pair<mlir::Value, mlir::Value> yield(mlir::Operation *op) {
    return {f(), g()};
  }

protected:
  std::function<mlir::Value()> f;
  std::function<mlir::Value()> g;
};

// Empty type constraint that accepts any previously inferred type
class NoTypeConstraint : public TypeConstraint {
public:
  void apply(mlir::Operation *op, TypeResolver &resolver,
             LocalInferenceState &currState,
             const LocalInferenceState &prevState) override {
    for (mlir::Value v : prevState.getValues()) {
      InferredType t = getFirstTypeInOrder(resolver, currState, prevState, v);
      currState.set(v, t);
    }
  }
};

// Base class for statically-defined type constraints, assuming `YieldT::yield`
// is a static function
template <typename YieldT> class StaticTypeConstraint : public TypeConstraint {
public:
  StaticTypeConstraint() = default;

protected:
  std::pair<mlir::Value, mlir::Value> yieldValues(mlir::Operation *op) {
    return YieldT::yield(op);
  }
};

// Base class for dynamically-defined type constraints, assuming `YieldT::yield`
// is a non-static function
template <typename YieldT> class DynamicTypeConstraint : public TypeConstraint {
public:
  template <typename... YieldTCtorArgTs>
  DynamicTypeConstraint(YieldTCtorArgTs &&...ctorArgs)
      : yield(std::forward<YieldTCtorArgTs>(ctorArgs)...) {}

protected:
  std::pair<mlir::Value, mlir::Value> yieldValues(mlir::Operation *op) {
    return yield.yield(op);
  }

  YieldT yield;
};

// Constraint that ensures that the types inferred for two values
// match exactly. If different types have been inferred previously for
// the two values, precedence is given to the type inferred for the
// first value yielded by `BaseConstraintT::yield`.
//
// If the types for the two values are fixed, but contradictory, an
// assertion is triggered.
template <typename BaseConstraintT>
class SameTypeConstraintBase : public BaseConstraintT {
public:
  using BaseConstraintT::BaseConstraintT;

  void apply(mlir::Operation *op, TypeResolver &resolver,
             LocalInferenceState &currState,
             const LocalInferenceState &prevState) override {
    auto [leftValue, rightValue] = this->yieldValues(op);

    mlir::Type leftType = leftValue.getType();
    mlir::Type rightType = rightValue.getType();

    bool leftUnresolved = resolver.isUnresolvedType(leftType);
    bool rightUnresolved = resolver.isUnresolvedType(rightType);

    // Priority is given from left to right, i.e., the type of the
    // first value yielded by `yield()` takes precendence over the
    // type for the second value
    if (!leftUnresolved && rightUnresolved) {
      currState.set(leftValue, leftType);
      currState.set(rightValue, leftType);
    } else if (leftUnresolved && !rightUnresolved) {
      currState.set(leftValue, rightType);
      currState.set(rightValue, rightType);
    } else if (!leftUnresolved && !rightUnresolved) {
      assert(leftType == rightType && "Constraint cannot be matched, as "
                                      "operands have different, fixed types");
      currState.set(leftValue, leftType);
      currState.set(rightValue, rightType);
    } else {
      // Both unresolved
      InferredType leftCurrType =
          this->getFirstTypeInOrder(resolver, currState, prevState, leftValue);
      InferredType rightCurrType =
          this->getFirstTypeInOrder(resolver, currState, prevState, rightValue);

      if (leftCurrType.hasType()) {
        currState.set(leftValue, leftCurrType);
        currState.set(rightValue, leftCurrType);
      } else if (rightCurrType.hasType()) {
        currState.set(leftValue, rightCurrType);
        currState.set(rightValue, rightCurrType);
      }
    }
  }
};

// Same type constraint for static yield operators
template <typename YieldT>
using SameTypeConstraint = SameTypeConstraintBase<StaticTypeConstraint<YieldT>>;

// Same type constraint for dynamic yield operators
template <typename YieldT>
using DynamicSameTypeConstraint =
    SameTypeConstraintBase<DynamicTypeConstraint<YieldT>>;

// Constraint that ensures that the types inferred for two values use
// the same element type. If different types have been inferred
// previously for the two values, precedence is given to the element
// type inferred for the first value yielded by
// `BaseConstraintT::yield`.
//
// If the element types for the two values are fixed, but
// contradictory, an assertion is triggered.
template <typename BaseConstraintT>
class SameElementTypeConstraintBase : public BaseConstraintT {
public:
  using BaseConstraintT::BaseConstraintT;

  void apply(mlir::Operation *op, TypeResolver &resolver,
             LocalInferenceState &currState,
             const LocalInferenceState &prevState) override {
    auto [leftValue, rightValue] = this->yieldValues(op);

    mlir::Type leftType = leftValue.getType();
    mlir::Type rightType = rightValue.getType();

    bool leftUnresolved = resolver.isUnresolvedType(leftType);
    bool rightUnresolved = resolver.isUnresolvedType(rightType);

    // Priority is given from left to right, i.e., the type of the
    // first value yielded by `yield()` takes precendence over the
    // type for the second value
    if (!leftUnresolved && rightUnresolved) {
      currState.set(leftValue, leftType);
      currState.set(rightValue, TypeInferenceUtils::applyElementType(
                                    getElementType(leftType), rightType));
    } else if (leftUnresolved && !rightUnresolved) {
      currState.set(leftValue, TypeInferenceUtils::applyElementType(
                                   getElementType(rightType), leftType));
      currState.set(rightValue, rightType);
    } else if (!leftUnresolved && !rightUnresolved) {
      assert(getElementType(leftType) == getElementType(rightType) &&
             "Constraint cannot be matched, as fixed types of the values"
             "are incompatible types");
      currState.set(leftValue, leftType);
      currState.set(rightValue, rightType);
    } else {
      // Both unresolved
      InferredType leftCurrType =
          this->getFirstTypeInOrder(resolver, currState, prevState, leftValue);
      InferredType rightCurrType =
          this->getFirstTypeInOrder(resolver, currState, prevState, rightValue);

      if (leftCurrType.hasType()) {
        currState.set(leftValue, leftCurrType);
        currState.set(rightValue,
                      TypeInferenceUtils::applyElementType(
                          getElementType(leftCurrType.getType()), rightType));
      } else if (rightCurrType.hasType()) {
        currState.set(leftValue,
                      TypeInferenceUtils::applyElementType(
                          getElementType(rightCurrType.getType()), leftType));
        currState.set(rightValue, rightCurrType);
      }
    }
  }

protected:
  static mlir::Type getElementType(mlir::Type t) {
    if (mlir::RankedTensorType rtt =
            llvm::dyn_cast<mlir::RankedTensorType>(t)) {
      return rtt.getElementType();
    } else if (mlir::MemRefType mrt = llvm::dyn_cast<mlir::MemRefType>(t)) {
      return mrt.getElementType();
    } else {
      return t;
    }
  }
};

// Same element type constraint for static yield operators
template <typename YieldT>
using SameElementTypeConstraint =
    SameElementTypeConstraintBase<StaticTypeConstraint<YieldT>>;

// Same element type constraint for dynamic yield operators
template <typename YieldT>
using DynamicSameElementTypeConstraint =
    SameElementTypeConstraintBase<DynamicTypeConstraint<YieldT>>;

// Type constraint ensuring that two operands with the statically-defined
// indexes `a` and `b` of an operation have the same type
template <int a, int b>
using SameOperandTypeConstraint = SameTypeConstraint<OperandValueYield<a, b>>;

// Type constraint ensuring that two results with the statically-defined
// indexes `a` and `b` of an operation have the same type
template <int a, int b>
using SameResultTypeConstraint = SameTypeConstraint<ResultValueYield<a, b>>;

// Type constraint ensuring that the operand with the
// statically-defined index `a` and the result with the
// statically-defined index `b` of an operation have the same type
template <int operandIdx, int resultIdx>
using SameOperandAndResultTypeConstraint =
    SameTypeConstraint<OperandAndResultValueYield<operandIdx, resultIdx>>;

// Type constraint ensuring that two operands with the statically-defined
// indexes `a` and `b` of an operation have the same element type
template <int a, int b>
using SameOperandElementTypeConstraint =
    SameElementTypeConstraint<OperandValueYield<a, b>>;

// Type constraint ensuring that two results with the statically-defined
// indexes `a` and `b` of an operation have the same element type
template <int a, int b>
using SameResultElementTypeConstraint =
    SameElementTypeConstraint<ResultValueYield<a, b>>;

// Type constraint ensuring that the operand with the
// statically-defined index `a` and the result with the
// statically-defined index `b` of an operation have the same element type
template <int operandIdx, int resultIdx>
using SameOperandAndResultElementTypeConstraint = SameElementTypeConstraint<
    OperandAndResultValueYield<operandIdx, resultIdx>>;

namespace {
namespace impl {
// Specialization needs to happen at namespace scope
template <typename ConstraintT, typename... ArgTs>
void addConstraint(std::vector<std::unique_ptr<TypeConstraint>> &constraints,
                   ArgTs &&...args) {
  std::unique_ptr<TypeConstraint> constraint =
      std::make_unique<ConstraintT>(std::forward<ArgTs>(args)...);
  constraints.push_back(std::move(constraint));
}

template <typename ConstraintT, typename... ArgTs>
void addConstraints(std::vector<std::unique_ptr<TypeConstraint>> &constraints,
                    ArgTs &&...args) {
  addConstraint<ConstraintT>(constraints, std::forward<ArgTs>(args)...);
}

template <typename ConstraintT0, typename ConstraintT1, typename... TailTs,
          typename... ArgTs>
void addConstraints(std::vector<std::unique_ptr<TypeConstraint>> &constraints,
                    ArgTs &&...args) {
  addConstraint<ConstraintT0>(constraints, std::forward<ArgTs>(args)...);
  addConstraints<ConstraintT1, TailTs..., ArgTs...>(
      constraints, std::forward<ArgTs>(args)...);
}

}; // namespace impl
}; // namespace

// Set of type constraints to be applied on one visit of an
// operation. The template parameter `maxApplications` sets a limit to
// the number of rounds the set of constraints is applied in
// `converge()`.
template <int maxApplications = 10> class TypeConstraintSet {
public:
  TypeConstraintSet() {}

  // Instantiates a constraint of the type `ConstraintT` with the
  // arguments `args` and adds the constraint to the set
  template <typename ConstraintT, typename... ArgTs>
  void addConstraint(ArgTs &&...args) {
    impl::addConstraint<ConstraintT, ArgTs...>(constraints,
                                               std::forward<ArgTs>(args)...);
  }

  // Instantiates the constraints of types specified by `ConstraintTs` with the
  // arguments `args` and adds them to the set
  template <typename... ConstraintTs, typename... ArgTs>
  void addConstraints(ArgTs &&...args) {
    impl::addConstraints<ConstraintTs..., ArgTs...>(
        constraints, std::forward<ArgTs>(args)...);
  }

  // Applies all type constraints in order of their addition for a
  // maximum of `maxApplications` rounds. Initial state needs to be
  // provided in `prevState`. The resulting state is contained in
  // `currState` afterward th call.
  //
  // If the rules converge on less or equal to `maxApplications`
  // rounds, the function return `true`, otherwise `false`.
  bool converge(mlir::Operation *op, TypeResolver &resolver,
                LocalInferenceState &currState,
                const LocalInferenceState &prevState) {
    // Initialize with fixed types
    for (mlir::Value v : op->getOperands()) {
      if (!resolver.isUnresolvedType(v.getType()))
        currState.set(v, v.getType());
    }

    for (mlir::Value v : op->getResults()) {
      if (!resolver.isUnresolvedType(v.getType()))
        currState.set(v, v.getType());
    }

    // Apply type constraints until state converges or maximum number
    // of iterations has been reached
    for (int i = 0; i < maxApplications; i++) {
      currState.setUnchanged();

      for (auto &constraint : constraints) {
        constraint->apply(op, resolver, currState, prevState);
      }

      if (!currState.hasChanged())
        return true;
    }

    return !currState.hasChanged();
  }

protected:
  std::vector<std::unique_ptr<TypeConstraint>> constraints;
};

// Base class for forward and backward type analysis with shared
// infrastructure. The template parameter `AnalysisT` must either be
// `mlir::dataflow::SparseDataFlowAnalysis` or
// `mlir::dataflow::SparseBackwardDataFlowAnalysis`.
template <typename AnalysisT>
class TypeInferenceAnalysisBase : public AnalysisT {
public:
  // Constructor passing on `constructorArgs` to the analysis class
  template <typename... ConstructorArgTs>
  TypeInferenceAnalysisBase(TypeResolver &resolver,
                            mlir::DataFlowSolver &solver,
                            ConstructorArgTs &&...constructorArgs)
      : AnalysisT(solver, std::forward<ConstructorArgTs>(constructorArgs)...),
        resolver(resolver) {}

protected:
  // Returns the types inferred for all values related to the
  // operation `op` from the respective lattice values. For values,
  // for which no type has been inferred so far, a new lattice with an
  // empty inferred types is created.
  LocalInferenceState getCurrentInferredTypes(mlir::Operation *op) {
    LocalInferenceState map =
        TypeInferenceUtils::getLocalInferenceState(op, [&](mlir::Value v) {
          return this->template getOrCreate<TypeInferenceLattice>(v);
        });

    return map;
  }

  // Dumps the inferred type for all values related to `op` on
  // `llvm::dbgs()`.
  void debugPrintOp(mlir::Operation *op) {
    LocalInferenceState inferredTypes = getCurrentInferredTypes(op);

    llvm::dbgs() << "Inference state for " << op->getName() << " \n";
    for (mlir::Value operand : op->getOperands()) {
      InferredType operandType = inferredTypes.find(operand);
      llvm::dbgs() << "  Operand: " << operandType << "\n";
    }

    for (mlir::Value result : op->getResults()) {
      InferredType resultType = inferredTypes.find(result);
      llvm::dbgs() << "  Result: " << resultType << "\n";
    }

    for (mlir::Region &r : op->getRegions()) {
      for (mlir::Value v : r.getArguments()) {
        InferredType argType = inferredTypes.find(v);
        llvm::dbgs() << "  RegionArg: " << argType << "\n";
      }
    }
  }

  // Updates the lattice for the inferred type for the value `v` using
  // its inferred type specified by `state`.
  void updateLatticeValuesFromState(LocalInferenceState &state, mlir::Value v) {
    TypeInferenceLattice *lattice =
        this->template getOrCreate<TypeInferenceLattice>(v);
    auto latticeValue = lattice->getValue();
    latticeValue.setType(state.find(v));
    mlir::ChangeResult res = lattice->join(latticeValue);
    this->propagateIfChanged(lattice, res);
  }

  // Visits a single operation: looks up the lattice values for the
  // values related to the operation, extracts the inferred types,
  // invokes the type resolver and updates the lattice values with the
  // newly inferred types.
  void doVisitOperation(Operation *op) {
    // Skip operations that do not use unresolved types
    if (TypeInferenceUtils::iterateRelatedValues(op, [&](mlir::Value v) {
          return !resolver.isUnresolvedType(v.getType());
        })) {
      return;
    }

    // Retrieve the current state of inference from the lattice values
    const LocalInferenceState inferredTypes = getCurrentInferredTypes(op);
    LocalInferenceState state;

    // Treat special op types for type inference
    mlir::TypeSwitch<mlir::Operation *>(op)
        .Case<TypeInference::PropagateUpwardOp>([&](auto op) {
          state.set(op->getOperand(0), op->getResult(0).getType());
          state.set(op->getResult(0), op->getResult(0).getType());
        })
        .template Case<TypeInference::PropagateDownwardOp>([&](auto op) {
          state.set(op->getOperand(0), op->getOperand(0).getType());
          state.set(op->getResult(0), op->getOperand(0).getType());
        })

        // Actually resolve the types based on the current state of
        // inference
        .Default([&](auto op) { state = resolver.resolve(op, inferredTypes); });

    // Store the resulting state in the lattice values
    TypeInferenceUtils::iterateRelatedValues(op, [&](mlir::Value v) {
      updateLatticeValuesFromState(state, v);
      return true;
    });
  }

  // Redirects the request for the initial inferred type of an
  // `mlir::Value` to the type resolver by invoking the resolver with
  // the owning operation of the value if it's a block argument.
  void doInitializeForValue(mlir::Value v) {
    if (mlir::BlockArgument ba = llvm::dyn_cast<mlir::BlockArgument>(v)) {
      doVisitOperation(ba.getOwner()->getParentOp());
    }
  }

  // Prints an indentation composed of `indent` times `" "`.
  void printIndent(int indent) {
    for (int i = 0; i < indent; i++)
      llvm::dbgs() << "  ";
  }

  // Dumps the state of type inference for the operation `op` with an
  // indentation level of `indent` as the name of the operation,
  // followed by the types inferred for each operand, followed by
  // `->`, followed by a dump of the state for any operation nested in
  // any region of `op`.
  void dumpStateForOp(mlir::Operation *op, int indent) {
    const LocalInferenceState state = getCurrentInferredTypes(op);

    printIndent(indent);
    llvm::dbgs() << op->getName() << ": (";
    for (mlir::Value v : op->getOperands()) {
      llvm::dbgs() << state.find(v) << ", ";
    }

    llvm::dbgs() << ") -> (";

    for (mlir::Value v : op->getResults()) {
      llvm::dbgs() << state.find(v) << ", ";
    }

    llvm::dbgs() << ")\n";

    for (mlir::Region &r : op->getRegions())
      for (mlir::Block &b : r.getBlocks())
        for (mlir::Operation &childOp : b.getOperations())
          dumpStateForOp(&childOp, indent + 1);
  }

  // Dumps the entire state of type inference for the function
  // containing the operation `op`. For each operation, this prints
  // the name of the operation, followed by the types inferred for
  // each operand, followed by `->`, followed by the types inferred
  // for the results.
  void dumpAllState(mlir::Operation *op) {
    mlir::Operation *funcOp = op;
    while (funcOp && !llvm::isa<mlir::func::FuncOp>(funcOp))
      funcOp = funcOp->getParentOp();

    assert(funcOp);

    dumpStateForOp(funcOp, 0);
  }

  TypeResolver &resolver;
};

// Type inference analysis running forward by following the flow of
// data from sources to sinks. For each operation encountered, the
// type resolver is invoked in order to update types until
// convergence. May be combined with `BackwardTypeInferenceAnalysis`
// for bidirectional type inference.
class ForwardTypeInferenceAnalysis
    : public TypeInferenceAnalysisBase<
          mlir::dataflow::SparseDataFlowAnalysis<TypeInferenceLattice>> {
public:
  ForwardTypeInferenceAnalysis(mlir::DataFlowSolver &solver,
                               TypeResolver &resolver)
      : TypeInferenceAnalysisBase(resolver, solver) {}

  void setToEntryState(TypeInferenceLattice *lattice) override {
    TypeInferenceAnalysisBase::doInitializeForValue(lattice->getPoint());
  }

  void
  visitOperation(Operation *op,
                 ArrayRef<const TypeInferenceLattice *> latticeOperandTypes,
                 ArrayRef<TypeInferenceLattice *> latticeResultTypes) override {
    TypeInferenceAnalysisBase::doVisitOperation(op);
  }
};

// Type inference analysis running backward by tracking back the flow
// of data from sinks to sources. For each operation encountered, the
// type resolver is invoked in order to update types until
// convergence. May be combined with `ForwardTypeInferenceAnalysis`
// for bidirectional type inference.
class BackwardTypeInferenceAnalysis
    : public TypeInferenceAnalysisBase<
          mlir::dataflow::SparseBackwardDataFlowAnalysis<
              TypeInferenceLattice>> {
public:
  BackwardTypeInferenceAnalysis(mlir::DataFlowSolver &solver,
                                mlir::SymbolTableCollection &symTabs,
                                TypeResolver &resolver)
      : TypeInferenceAnalysisBase(resolver, solver, symTabs) {}

  void setToExitState(TypeInferenceLattice *lattice) override {
    TypeInferenceAnalysisBase::doInitializeForValue(lattice->getPoint());
  }

  void visitOperation(
      Operation *op, ArrayRef<TypeInferenceLattice *> latticeOperandTypes,
      ArrayRef<const TypeInferenceLattice *> latticeResultTypes) override {
    TypeInferenceAnalysisBase::doVisitOperation(op);
  }

  void visitBranchOperand(mlir::OpOperand &operand) override {}
};

} // namespace concretelang
} // namespace mlir

#endif
