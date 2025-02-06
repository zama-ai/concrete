// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <variant>

template <typename V> size_t hashVariant(const V &v);
template <typename V, typename T, typename... OtherTs>
size_t hashVariant(const V &v);
size_t hashUint64(uint64_t u);
template <typename T> size_t hashSmallVector(const llvm::SmallVector<T> &v);

namespace llvm {
// Extra hashing functions; these must be declared before any header
// that directly or indirectly includes `llvm/ADT/Hashing.h` is
// included. Unfortunately, this makes `llvm::hash_code` unavailable
// for the functions below. However, since implicit construction of
// `llvm::hash_code` is possible from `size_t`, these functions simply
// return `size_t`.
//
// The actual hashing is delegated to another function
// forward-declared above, but implemented below. This is necessary,
// because the basic hashing functions (e;g., for integers) are not
// available before the inclusion of `Hashing.h`.

// Hashing of std::variant
template <typename T, typename... OtherTs>
size_t hash_value(const std::variant<T, OtherTs...> &v) {
  return hashVariant<std::variant<T, OtherTs...>, T, OtherTs...>(v);
}

// Hashing of double
size_t hash_value(const double &d) {
  static_assert(sizeof(double) == sizeof(uint64_t),
                "Only implemented for 64-bit doubles");

  const uint64_t *u = reinterpret_cast<const uint64_t *>(&d);
  return hashUint64(*u);
}

// Hashing of SmallVector<T>
template <typename T> size_t hash_value(const llvm::SmallVector<T> &v) {
  return hashSmallVector(v);
}

} // namespace llvm

#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.h"
#include "concretelang/Dialect/GLWE/IR/GLWEDialect.h"
#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Dialect/GLWE/IR/GLWETypes.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "mlir/IR/DialectImplementation.h"
// #include "mlir/lib/AsmParser/AsmParserImpl.h"

// #include "concretelang/Dialect/GLWE/Interfaces/GLWEInterfaces.h"

// Terminal case for the recursion of `hashVariant` on the alternative
// types of the variant. Cannot happen if variant holds a value.
template <typename V> size_t hashVariant(const V &v) {
  llvm_unreachable("Variant without value");
  return 0;
}

// Recursion on the alternative types of the variant. Delegates to
// `llvm::hash_value` for the actual value.
template <typename V, typename T, typename... OtherTs>
size_t hashVariant(const V &v) {
  if (std::holds_alternative<T>(v))
    return ::llvm::hash_value(std::get<T>(v));
  else
    return hashVariant<V, OtherTs...>(v);
}

size_t hashUint64(uint64_t u) { return ::llvm::hash_value(u); }

template <typename T> size_t hashSmallVector(const llvm::SmallVector<T> &v) {
  ::llvm::hash_code h = 0;

  for (const T &e : v)
    h = ::llvm::hash_combine(h, e);

  return h;
}

namespace mlir {
template <> struct FieldParser<::mlir::concretelang::GLWE::GLWESymbolName> {
  static FailureOr<::mlir::concretelang::GLWE::GLWESymbolName>
  parse(AsmParser &parser) {
    mlir::StringAttr symbol;

    if (parser.parseSymbolName(symbol))
      return failure();

    return ::mlir::concretelang::GLWE::GLWESymbolName{symbol};
  }
};

namespace {
template <typename V> static FailureOr<V> tryParseVariant(AsmParser &parser) {
  return failure();
}

inline ::llvm::hash_code hash_value(const double &d) {
  static_assert(sizeof(double) == sizeof(uint64_t),
                "Hashing of doubles only supported for 64-bit double values");

  const uint64_t *uintptr = reinterpret_cast<const uint64_t *>(&d);
  return ::llvm::hash_value(*uintptr);
}

template <typename T, typename = T> struct OptionalFieldParser;

// Optional field parser for `double`. Parses either an explicit
// floating point literal or an integer literal. In case of an integer
// literal, the funcion checks if the value can be represented exactly
// as a `double`.
template <> struct OptionalFieldParser<double> {
  static FailureOr<double> parse(AsmParser &parser) {
    double d;
    int64_t i;

    if (parser.parseOptionalFloat(d).succeeded()) {
      return d;
    } else if (parser.parseOptionalInteger(i).has_value()) {
      if (static_cast<int64_t>(static_cast<double>(i)) != i) {
        return parser.emitError(parser.getCurrentLocation(), "Value ")
               << i << " cannot be represented as a double";
      } else {
        return static_cast<double>(i);
      }
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "Invalid floating point value");
    }
  }
};

// Same as `FieldParser<T>` for integral types `T`, but does not emit
// errors if no integer can be parsed.
template <typename IntT>
struct OptionalFieldParser<
    IntT, std::enable_if_t<std::is_integral<IntT>::value, IntT>> {
  static FailureOr<IntT> parse(AsmParser &parser) {
    IntT value;
    if (parser.parseOptionalInteger(value).has_value())
      return value;
    return failure();
  }
};

// Same as `FieldParser<T>` for vector types `T`, but does not emit
// errors if no list of values can be parsed.
template <typename ContainerT>
struct OptionalFieldParser<
    ContainerT, std::enable_if_t<llvm::is_detected<detail::has_push_back_t,
                                                   ContainerT>::value,
                                 ContainerT>> {
  using ElementT = typename ContainerT::value_type;
  static FailureOr<ContainerT> parse(AsmParser &parser) {
    ContainerT elements;
    auto elementParser = [&]() {
      auto element = OptionalFieldParser<ElementT>::parse(parser);
      if (failed(element))
        return failure();
      elements.push_back(std::move(*element));
      return success();
    };
    if (parser.parseCommaSeparatedList(elementParser))
      return failure();
    return elements;
  }
};

// Attempts to parse a value for the alternative type `TÀ` of a
// variant `V`. If this fails, the function continues with
// `TBs...`. Returns an error if none of the types `TA` or
// `TBs...` can be parsed.
template <typename V, typename TA, typename... TBs>
static FailureOr<V> tryParseVariant(AsmParser &parser) {
  parser.pushLexerPos();
  auto elem = OptionalFieldParser<TA>::parse(parser);

  if (failed(elem)) {
    parser.popLexerPos();
    return tryParseVariant<V, TBs...>(parser);
  } else {
    parser.popLexerPos(true);
    return V{*elem};
  }
}
} // namespace

// Field parser for Variants. Attempts to parse the data types of the
// variant in order of the variant's template type arguments.
template <typename T1, typename... OtherTs>
struct FieldParser<std::variant<T1, OtherTs...>> {
  using V = std::variant<T1, OtherTs...>;

  static FailureOr<V> parse(AsmParser &parser) {
    return tryParseVariant<V, T1, OtherTs...>(parser);
  }
};

namespace {
// Terminal case for printing a variant. Cannot be reached if the
// variant holds a value.
template <typename V>
mlir::AsmPrinter &printVariant(mlir::AsmPrinter &p, const V &v) {
  llvm_unreachable("Variant without value");
  return p;
}

// Idirection to force printing of vectors as `ArrayRef<T>` instead of
// `T*`.
template <typename T,
          std::enable_if_t<!std::is_constructible<
              llvm::ArrayRef<typename T::value_type>, T>::value> * = nullptr>
mlir::AsmPrinter &printCheckArrayRef(mlir::AsmPrinter &p, const T &v) {
  return p << v;
}

template <typename T,
          std::enable_if_t<std::is_constructible<
              llvm::ArrayRef<typename T::value_type>, T>::value> * = nullptr>
mlir::AsmPrinter &printCheckArrayRef(mlir::AsmPrinter &p, const T &v) {
  llvm::ArrayRef<typename T::value_type> ar = v;
  return p << ar;
}

// Prints a variant
template <typename V, typename T1, typename... OtherTs>
mlir::AsmPrinter &printVariant(mlir::AsmPrinter &p, const V &v) {
  if (auto pval = std::get_if<T1>(&v)) {
    return printCheckArrayRef(p, *pval);
  } else
    return printVariant<V, OtherTs...>(p, v);
}
} // namespace

template <typename T1, typename... OtherTs>
mlir::AsmPrinter &operator<<(mlir::AsmPrinter &p,
                             const std::variant<T1, OtherTs...> &v) {
  return printVariant<std::variant<T1, OtherTs...>, T1, OtherTs...>(p, v);
}
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWETypes.cpp.inc"

#define GET_OP_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEOps.cpp.inc"

#include "concretelang/Dialect/GLWE/IR/GLWEDialect.cpp.inc"

#include "concretelang/Support/Constants.h"

using namespace mlir::concretelang::GLWE;

void GLWEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/GLWE/IR/GLWEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/GLWE/IR/GLWETypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.cpp.inc"
      >();
}

// GLWEExpr ///////////////////////////////////////

::mlir::Attribute
mlir::concretelang::GLWE::GLWEExprAttr::parse(::mlir::AsmParser &parser,
                                              ::mlir::Type odsType) {
  // parse '<'
  if (parser.parseLess())
    return {};

  GLWEExpr result = GLWEExpr::parse(parser);

  // parse '>'
  if (parser.parseGreater())
    return {};
  return GLWEExprAttr::get(parser.getContext(), result);
}

void mlir::concretelang::GLWE::GLWEExprAttr::print(
    ::mlir::AsmPrinter &printer) const {
  printer << "<";
  this->getExpr().print(printer);
  printer << ">";
}
