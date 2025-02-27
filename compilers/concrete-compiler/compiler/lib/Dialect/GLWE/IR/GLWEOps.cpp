#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Common/Error.h"

#include <numeric>

namespace mlir {
namespace concretelang {
namespace GLWE {

using ::concretelang::error::Result;
using ::concretelang::error::StringError;

Result<GLWEExpr> averageMean(SecretKeyDistribution skDistribution,
                             mlir::MLIRContext *context) {
  switch (skDistribution) {
  case SecretKeyDistribution::Binary:
    return getGlweConstantExpr(0.5, context);
  case SecretKeyDistribution::Ternary:
    return getGlweConstantExpr(0, context);
  }
  llvm::llvm_unreachable_internal("Unknow secret key distribution");
}

Result<GLWEExpr> averageVariance(SecretKeyDistribution skDistribution,
                                 mlir::MLIRContext *context) {
  switch (skDistribution) {
  case SecretKeyDistribution::Binary:
    return getGlweConstantExpr(0.25, context);
  case SecretKeyDistribution::Ternary:
    return getGlweConstantExpr(2. / 3., context);
  }
  llvm::llvm_unreachable_internal("Unknow secret key distribution");
}

Result<GLWEExpr> undot(llvm::StringRef dots, GLWESizeAttr size) {
  auto [field, rest] = dots.split(".");
  if (field == "dimension") {
    return size.getDimension().getExpr();
  }
  if (field == "poly_size") {
    return size.getPolySize().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.size attribute";
}

Result<GLWEExpr> undot(llvm::StringRef dots,
                       SecretKeyDistributionAttr skDistributionAttr) {
  auto [field, rest] = dots.split(".");
  auto skDistribution = skDistributionAttr.getValue();
  auto context = skDistributionAttr.getContext();
  if (field == "average_mean") {
    return averageMean(skDistribution, context);
  }
  if (field == "average_variance") {
    return averageVariance(skDistribution, context);
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.secret_key_distribution attribute";
}

Result<GLWEExpr> undot(llvm::StringRef dots, GLWESecretKeyAttr sk) {
  auto [field, rest] = dots.split(".");
  if (field == "size") {
    return undot(rest, sk.getSize());
  }
  if (field == "nb_keys") {
    return sk.getNbKeys().getExpr();
  }
  if (field == "distribution") {
    return undot(rest, sk.getDistribution());
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.secret_key attribute";
}

Result<GLWEExpr> undot(llvm::StringRef dots, GLWEEncodingAttr sk) {
  auto [field, rest] = dots.split(".");
  if (field == "body_modulus") {
    return sk.getBodyModulus().getExpr();
  }
  if (field == "mask_modulus") {
    return sk.getMaskModulus().getExpr();
  }
  if (field == "message_modulus") {
    return sk.getMessageModulus().getExpr();
  }
  if (field == "right_padding") {
    return sk.getRightPadding().getExpr();
  }
  if (field == "left_padding") {
    return sk.getLeftPadding().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.encoding attribute";
}

Result<GLWEExpr> undot(llvm::StringRef dots, DecompositionParametersAttr sk) {
  auto [field, rest] = dots.split(".");
  if (field == "base") {
    return sk.getBase().getExpr();
  }
  if (field == "level") {
    return sk.getLevel().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.decomposition attribute";
}

Result<GLWEExpr> undot(llvm::StringRef dots,
                       llvm::ArrayRef<GLWEExprAttr> array) {
  auto [field, rest] = dots.split(".");
  if (field == "last") {
    return array.back().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for shape attribute";
}

Result<GLWEExpr> undot(llvm::StringRef dots, GLWEType glwe) {
  auto [field, rest] = dots.split(".");
  if (field == "secret_key") {
    return undot(rest, glwe.getSecretKey());
  }
  if (field == "encoding") {
    return undot(rest, glwe.getEncoding());
  }
  if (field == "variance") {
    return glwe.getVariance().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.glwe type";
}

Result<GLWEExpr> undot(llvm::StringRef dots, RadixGLWEType glwe) {
  auto [field, rest] = dots.split(".");
  if (field == "secret_key") {
    return undot(rest, glwe.getSecretKey());
  }
  if (field == "encoding") {
    return undot(rest, glwe.getEncoding());
  }
  if (field == "decomposition") {
    return undot(rest, glwe.getDecomposition());
  }
  if (field == "variance") {
    return glwe.getVariance().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.radix_glwe type";
}

Result<GLWEExpr> undot(llvm::StringRef dots, GLevType glwe) {
  auto [field, rest] = dots.split(".");
  if (field == "secret_key") {
    return undot(rest, glwe.getSecretKey());
  }
  if (field == "encoding") {
    return undot(rest, glwe.getEncoding());
  }
  if (field == "decomposition") {
    return undot(rest, glwe.getDecomposition());
  }
  if (field == "average_message") {
    auto expr = glwe.getAverageMessage().getExpr();
    if (auto symbol = expr.dyn_cast<GlweSymbolExpr>()) {
      auto [field, rest] = symbol.getSymbolName().split(".");
      if (field == "self") {
        return undot(rest, glwe);
      }
    }
    return expr;
  }
  if (field == "shape") {
    return undot(rest, glwe.getShape());
  }
  if (field == "variance") {
    return glwe.getVariance().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.glev type";
}

template <typename Op> Result<GLWEExpr> _resolveOutputVariance(Op *op) {
  if (!op->getVariance().has_value()) {
    return StringError("no variance formula to resolve");
  }
  std::optional<::concretelang::error::StringError> err;
  auto replacement = [&](GLWEExpr e) {
    if (auto symbolExpr = e.dyn_cast<GlweSymbolExpr>()) {
      auto [field, rest] = symbolExpr.getSymbolName().split(".");
      if (field != "self")
        return e;
      auto u = undot(rest, *op);
      if (u.has_error()) {
        err = u.error();
        return e;
      }
      return u.value();
    }
    return e;
  };
  auto variance = op->getVariance().value().getExpr().replace(replacement);
  if (err.has_value())
    return err.value();

  return variance;
}

/////////////////////////////////////////////////
// GenericOP ////////////////////////////////////

Result<GLWEExpr> undot(llvm::StringRef dots, GenericOP op) {
  auto [field, rest] = dots.split(".");
  if (field.starts_with("in")) {
    llvm::APInt pos;
    auto suffix = field.substr(2);
    if (suffix.getAsInteger(10, pos)) {
      return StringError("unexpected field(")
             << field.str() << ") for glwe.generic op the suffix("
             << suffix.str() << ") should be a integer";
    }
    auto i = pos.getZExtValue();
    if (i >= op.getNumOperands()) {
      return StringError("unexped field(")
             << field.str() << ") glwe.generic as only " << op.getNumOperands()
             << " operands";
    }
    auto type = op.getOperandTypes()[pos.getZExtValue()];
    if (auto glwe = type.dyn_cast<GLWEType>()) {
      return undot(rest, glwe);
    }
    if (auto radix = type.dyn_cast<RadixGLWEType>()) {
      return undot(rest, radix);
    }
    if (auto glev = type.dyn_cast<GLevType>()) {
      return undot(rest, glev);
    }
    return StringError("unexpected field(")
           << field.str() << ") for glwe.generic op type not supported";
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.moduls_switch op";
}

Result<GLWEExpr> GenericOP::resolveOutputVariance() {
  return _resolveOutputVariance<GenericOP>(this);
}

GLWEExpr GenericOP::defaultVariance() {
  return getGlweConstantExpr(0., getContext());
}

/////////////////////////////////////////////////
// Dot //////////////////////////////////////////

mlir::LogicalResult Dot::verify() {
  auto inputType = this->getInput()
                       .getType()
                       .dyn_cast<TensorType>()
                       .getElementType()
                       .dyn_cast<GLWEType>();
  auto outputType = this->getOutput().getType().dyn_cast<GLWEType>();
  if (inputType.getSecretKey() != outputType.getSecretKey()) {
    this->emitOpError(
        "input and output glwe.glwe must share the same secret_key "
        "attribute");
    return mlir::failure();
  }
  if (inputType.getEncoding() != outputType.getEncoding()) {
    this->emitOpError("input and output glwe.glwe must share the same encoding "
                      "attribute");
    return mlir::failure();
  }
  if ((size_t)this->getInput().getType().dyn_cast<TensorType>().getDimSize(0) !=
      this->getWeights().size()) {
    this->emitOpError("The dimension of the input 1D tensor should be equals "
                      "to the number of weights");
    return mlir::failure();
  }
  return mlir::success();
}

Result<GLWEExpr> undot(llvm::StringRef dots, Dot op) {
  auto [field, rest] = dots.split(".");
  if (field == "input") {
    return undot(rest, op.getInput()
                           .getType()
                           .dyn_cast<TensorType>()
                           .getElementType()
                           .dyn_cast<GLWEType>());
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.moduls_switch op";
}

Result<GLWEExpr> Dot::resolveOutputVariance() {
  return _resolveOutputVariance<Dot>(this);
}

Result<GLWEExpr> attrToExpr(mlir::Attribute attr) {
  if (auto expr = attr.dyn_cast<GLWEExprAttr>()) {
    return expr.getExpr();
  }
  if (auto int_ = attr.dyn_cast<IntegerAttr>()) {
    return getGlweConstantExpr(int_.getValue().getSExtValue(),
                               attr.getContext());
  }
  if (auto int_ = attr.dyn_cast<FloatAttr>()) {
    return getGlweConstantExpr(int_.getValue().convertToDouble(),
                               attr.getContext());
  }
  return StringError("attribute cannot be converted to glwe.expr");
}

GLWEExpr Dot::defaultVariance() {
  auto var = getGlweConstantExpr(0, getContext());
  for (auto x : llvm::enumerate(this->getWeights())) {
    auto weightExpr = attrToExpr(x.value());
    if (weightExpr.has_failure()) {
      assert(false);
    }
    var = var + weightExpr.value();
  }
  var = var * getGlweSymbolExpr("self.input.variance", getContext());
  return var;
}

mlir::LogicalResult ModulusSwitch::verify() {
  auto input = this->getInput().getType().dyn_cast<GLWEType>();
  auto inputSK = input.getSecretKey();
  auto output = this->getOutput().getType().dyn_cast<GLWEType>();
  auto outputSK = output.getSecretKey();
  if (inputSK != outputSK) {
    this->emitOpError("failed to verify that the input and output {secret_key} "
                      "parameters are equals");
    return mlir::failure();
  }

  auto inputEncoding = input.getEncoding();
  auto outputEncoding = output.getEncoding();
  if (inputEncoding.getMessageModulus() != outputEncoding.getMessageModulus() ||
      inputEncoding.getRightPadding() != outputEncoding.getRightPadding() ||
      inputEncoding.getLeftPadding() != outputEncoding.getLeftPadding() ||
      ((!this->getBody()) &&
       inputEncoding.getBodyModulus() != outputEncoding.getBodyModulus()) ||
      ((!this->getMask()) &&
       inputEncoding.getMaskModulus() != outputEncoding.getMaskModulus())) {
    this->emitOpError("failed to verify that the input and output "
                      "{encoding} parameters matches");
    return mlir::failure();
  }
  if (this->getBody() &&
      this->getModulus() != outputEncoding.getBodyModulus()) {
    this->emitOpError("failed to verify that the {modulus} and output "
                      "{encoding.body_modulus} "
                      "parameters are equals");
    return mlir::failure();
  }
  if (this->getMask() &&
      this->getModulus() != outputEncoding.getMaskModulus()) {
    this->emitOpError("failed to verify that the {modulus} and output "
                      "{encoding.mask_modulus} "
                      "parameters are equals");
    return mlir::failure();
  }
  return mlir::success();
}

GLWEExpr ModulusSwitch::defaultVariance() {
  auto input = this->getInput().getType().dyn_cast<GLWEType>();
  auto oldModulusAttr = input.getEncoding().getBodyModulus();
  if (!this->getBody()) {
    oldModulusAttr = input.getEncoding().getMaskModulus();
  }
  auto oldModulusExpr = oldModulusAttr.getExpr().simplify();
  auto modulusExpr = this->getModulus().getExpr().simplify();
  auto lcm = oldModulusExpr;
  if (oldModulusExpr.isa<GlweConstantExpr>() &&
      modulusExpr.isa<GlweConstantExpr>()) {
    auto oldModulus = oldModulusExpr.dyn_cast<GlweConstantExpr>().getValue();
    assert(oldModulus == trunc(oldModulus));
    assert(modulusExpr.isa<GlweConstantExpr>());
    auto modulus = oldModulusExpr.dyn_cast<GlweConstantExpr>().getValue();
    assert(modulus == trunc(modulus));
    auto _lcm = std::lcm((uint64_t)oldModulus, (uint64_t)modulus);
    lcm = getGlweConstantExpr(_lcm, getContext());
  }
  // TODO : add lcm, odd_ratio, and randomized_rounding parameter
  auto odd_ratio = false;
  auto randomized_rounding = false;

  auto glweDim =
      getGlweSymbolExpr("self.input.secret_key.size.dimension", getContext());
  auto polySize =
      getGlweSymbolExpr("self.input.secret_key.size.poly_size", getContext());
  auto lweDim = glweDim * polySize;
  auto averageSkMean = getGlweSymbolExpr(
      "self.input.secret_key.distribution.average_mean", getContext());
  auto averageSkMeanSquared = averageSkMean.pow(2);
  auto averageSkVariance = getGlweSymbolExpr(
      "self.input.secret_key.distribution.average_variance", getContext());
  auto vOdd = (1 / modulusExpr.pow(2) - 1 / lcm.pow(2)) / 12;
  auto vEven = (1 / modulusExpr.pow(2) + 2 / lcm.pow(2)) / 12;
  GLWEExpr vBody;
  if (odd_ratio) {
    vBody = vOdd;
  } else {
    vBody = vEven;
  }
  GLWEExpr vMask;
  if (randomized_rounding || odd_ratio) {
    vMask = lweDim * vBody * (averageSkMeanSquared + averageSkVariance);
  } else {
    vMask = lweDim * (vEven * averageSkMeanSquared + vOdd * averageSkVariance);
  }
  if (this->getBody() && this->getMask()) {
    return vMask + vBody;
  }
  if (this->getBody()) {
    return vBody;
  }
  return vMask;
}

Result<GLWEExpr> undot(llvm::StringRef dots, ModulusSwitch op) {
  auto [field, rest] = dots.split(".");
  if (field == "input") {
    return undot(rest, op.getInput().getType().dyn_cast<GLWEType>());
  }
  if (field == "modulus") {
    return op.getModulus().getExpr();
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.moduls_switch op";
}

Result<GLWEExpr> ModulusSwitch::resolveOutputVariance() {
  return _resolveOutputVariance<ModulusSwitch>(this);
}

mlir::LogicalResult ExactDecompose::verify() {
  auto input = this->getInput().getType().dyn_cast<GLWEType>();
  auto inputSK = input.getSecretKey();
  auto output = this->getOutput().getType().dyn_cast<RadixGLWEType>();
  auto outputSK = output.getSecretKey();
  if (inputSK != outputSK) {
    this->emitOpError("failed to verify that the input and output {secret_key} "
                      "parameters are equals");
    return mlir::failure();
  }
  auto inputEncoding = input.getEncoding();
  auto outputEncoding = output.getEncoding();
  if (inputEncoding != outputEncoding) {
    this->emitOpError("failed to verify that the input and output {encoding} "
                      "parameters are equals");
    return mlir::failure();
  }
  if (output.getDecomposition() != this->getDecomposition()) {
    this->emitOpError("failed to verify that the op and output {decomposition} "
                      "parameters are equals");
    return mlir::failure();
  }
  if (output.getBody() != this->getBody()) {
    this->emitOpError("failed to verify that the op and output {body} "
                      "parameters are equals");
    return mlir::failure();
  }
  if (output.getMask() != this->getMask()) {
    this->emitOpError("failed to verify that the op and output {body} "
                      "parameters are equals");
    return mlir::failure();
  }
  return mlir::success();
}

GLWEExpr ExactDecompose::defaultVariance() {
  return getGlweSymbolExpr("self.input.variance", getContext());
}

Result<GLWEExpr> undot(llvm::StringRef dots, ExactDecompose op) {
  auto [field, rest] = dots.split(".");
  if (field == "input") {
    return undot(rest, op.getInput().getType().dyn_cast<GLWEType>());
  }
  if (field == "decomposition") {
    return undot(rest, op.getDecomposition());
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.exact_decompose op";
}

Result<GLWEExpr> ExactDecompose::resolveOutputVariance() {
  return _resolveOutputVariance<ExactDecompose>(this);
}

mlir::LogicalResult ExactRecompose::verify() {
  auto input = this->getInput().getType().dyn_cast<RadixGLWEType>();
  auto inputSK = input.getSecretKey();
  auto glev = this->getGlev().getType().dyn_cast<GLevType>();
  auto glevSize = glev.getShape().back();
  auto skDim = inputSK.getSize().getDimension().getExpr();
  if (input.getBody()) {
    if (skDim + inputSK.getNbKeys().getExpr() != glevSize.getExpr()) {
      this->emitOpError("failed to verify that the glev size is equals to the "
                        "input secret key "
                        "GLWE dimension + nb_keys as body is {true}");
      return mlir::failure();
    }
  } else if (skDim != glevSize.getExpr()) {
    this->emitOpError("failed to verify that the glev size is equals to the "
                      "input secret key "
                      "GLWE dimension ");
    return mlir::failure();
  }
  if (glev.getDecomposition() != input.getDecomposition()) {
    this->emitOpError("failed to verify that the glev and input "
                      "{decomposition} parameters are equals");
    return mlir::failure();
  }
  auto glevEncoding = glev.getEncoding();
  auto inputEncoding = input.getEncoding();
  if (!input.getBody() &&
      inputEncoding.getBodyModulus() != glevEncoding.getBodyModulus()) {
    this->emitOpError("failed to verify that the input and glev "
                      "{encoding.body_modulus} are "
                      "equals as the input {body} parameter is false");
    return mlir::failure();
  }
  if (glevEncoding.getMaskModulus() != glevEncoding.getBodyModulus()) {
    this->emitOpError("failed to verify that the glev {encoding.body_modulus} "
                      "is equal to glev {encoding.mask_modulus}");
    return mlir::failure();
  }
  auto output = this->getOutput().getType().dyn_cast<GLWEType>();
  auto expectedEncoding = GLWEEncodingAttr::get(
      getContext(), glevEncoding.getMaskModulus(),
      glevEncoding.getBodyModulus(), inputEncoding.getMessageModulus(),
      inputEncoding.getRightPadding(), inputEncoding.getLeftPadding());
  if (output.getEncoding() != expectedEncoding) {
    this->emitOpError(
        "failed to verify that output encoding match the expected encoding. "
        "The output encoding must be equal to the input encoding with "
        "{mask_modulus, body_modulus} parameter equal to the {mask_modulus} "
        "of "
        "the glev encoding");
    return mlir::failure();
  }
  if (glev.getSecretKey() != output.getSecretKey()) {
    this->emitOpError("failed to verify that the glev and output {secret_key} "
                      "parameters are equals");
    return mlir::failure();
  }

  return mlir::success();
}

GLWEExpr fft_var_base(GLWEExpr base, GLWEExpr modulus, int mantissa) {
  auto bits_lost = max(0, modulus.log2() - mantissa);
  return getGlweConstantExpr(2, base.getContext())
      .pow(base.log2() * 2 + bits_lost * 2 - modulus.log2() * 2);
}

GLWEExpr fft_variance(GLWEExpr glevPolySize, GLWEExpr glevDimension,
                      GLWEExpr glevLevel, GLWEExpr glevBase,
                      GLWEExpr glevModulus, int mantissa, int groupingFactor,
                      GLWEExpr nbBodies) {
  auto fftBase = fft_var_base(glevBase, glevModulus, mantissa);
  std::vector<double> coefficients;
  switch (groupingFactor) {
  case 1: // unclear why no gap is visible around log-B-bound
    coefficients = {0.00705, 1.22003, 1.01827}; // post-bound
    break;
  case 2:
    coefficients = {0.00220, 1.94548, 1.04148}; // pre-bound
    break;
  case 3:
    coefficients = {0.00492, 1.90722, 1.01110}; // pre-bound
    // {0.004919593456537255, 1.9072214291896599, 1.011101247104754}
    break;
  case 4:
    coefficients = {0.00855, 1.90759, 1.00715}; // pre-bound
    break;
  default:
    assert(false && "Grouping factor  not supported !!");
  }
  return fftBase * coefficients[0] *
         (glevDimension * glevPolySize).pow(coefficients[1]) *
         (glevLevel * (glevDimension + nbBodies)).pow(coefficients[2]) *
         glevPolySize;
}

GLWEExpr ExactRecompose::defaultVariance() {
  auto numberOfNonZeroCoefs =
      getGlweSymbolExpr("self.input.secret_key.size.poly_size", getContext());
  if (auto attr = this->getInput()
                      .getType()
                      .dyn_cast<RadixGLWEType>()
                      .getNumberOfNonZerosCoefs()) {
    numberOfNonZeroCoefs = attr.getExpr();
  }

  auto inputVariance = getGlweSymbolExpr("self.input.variance", getContext());
  auto glevVariance = getGlweSymbolExpr("self.glev.variance", getContext());
  auto glevLevel =
      getGlweSymbolExpr("self.glev.decomposition.level", getContext());
  auto glevBase =
      getGlweSymbolExpr("self.glev.decomposition.base", getContext());
  auto glevSize = getGlweSymbolExpr("self.glev.shape.last", getContext());
  auto glevAverageMessage =
      getGlweSymbolExpr("self.glev.average_message", getContext());
  // exact recomposition variance
  auto bound = glevBase / 2;
  auto averageCoeff = (bound.pow(2) + 1. / 2.) / 3;
  auto coeffGlevVariance =
      glevSize * glevLevel * numberOfNonZeroCoefs * averageCoeff;
  auto recomposeVariance =
      coeffGlevVariance * glevVariance + inputVariance * glevAverageMessage;
  if (this->getFft()) {
    auto glevPolySize =
        getGlweSymbolExpr("self.glev.secret_key.size.poly_size", getContext());
    auto glevGlweDim =
        getGlweSymbolExpr("self.glev.secret_key.size.dimension", getContext());
    auto mantissa = getMantissaSize();
    auto groupingFactor = getGroupingFactor();
    auto glevModulus =
        getGlweSymbolExpr("self.glev.encoding.mask_modulus", getContext());
    auto nbBodies =
        getGlweSymbolExpr("self.input.secret_key.nb_keys", getContext());
    recomposeVariance =
        recomposeVariance + fft_variance(glevPolySize, glevGlweDim, glevLevel,
                                         glevBase, glevModulus, mantissa,
                                         groupingFactor, nbBodies);
  }
  return recomposeVariance;
}

Result<GLWEExpr> undot(llvm::StringRef dots, ExactRecompose op) {
  auto [field, rest] = dots.split(".");
  if (field == "input") {

    return undot(rest, op.getInput().getType().dyn_cast<RadixGLWEType>());
  }
  if (field == "glev") {
    return undot(rest, op.getGlev().getType().dyn_cast<GLevType>());
  }
  if (field == "grouping_factor") {
    return getGlweConstantExpr(op.getGroupingFactor(), op.getContext());
  }
  if (field == "mantissa_size") {
    return getGlweConstantExpr(op.getMantissaSize(), op.getContext());
  }
  return StringError("unexpected field(")
         << field.str() << ") for glwe.exact_recompose op";
}

Result<GLWEExpr> ExactRecompose::resolveOutputVariance() {
  return _resolveOutputVariance<ExactRecompose>(this);
}
} // namespace GLWE
} // namespace concretelang
} // namespace mlir