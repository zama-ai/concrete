#include "concretelang/Dialect/GLWE/IR/GLWETypes.h"
#include "concretelang/Common/Security.h"
#include <numeric>

namespace mlir {
namespace concretelang {
namespace GLWE {

// GLWEExpr minimalVariance(GLWEExpr modulus, GLWEExpr size) {
//   auto curve = ::concretelang::security::getSecurityCurve();
// }

::concretelang::security::KeyFormat
securityDistributionToNoiseFormat(SecretKeyDistributionKind skDistribution) {
  switch (skDistribution) {
  case SecretKeyDistributionKind::Binary:
    return ::concretelang::security::KeyFormat::BINARY;
  default:
    break;
  }
  assert(false && "SecretKeyDistribution not supported");
}

double getEpsilonBits(NoiseDistribution noiseDistribution) {
  switch (noiseDistribution) {
  case NoiseDistribution::Gaussian:
    return 2;
  case NoiseDistribution::TUniform:
    return 2.2;
  default:
    break;
  }
  assert(false && "NoiseDistribution not supported");
}

GLWEExpr
unsafeMinimalVarianceLog2(::concretelang::security::SecurityCurve curve,
                          NoiseDistribution noiseDistribution, GLWEExpr size,
                          GLWEExpr modulus) {
  switch (noiseDistribution) {
  case NoiseDistribution::Gaussian: {
    auto secure_log2_std_size = curve.slope * size;
    auto secure_log2_std = secure_log2_std_size + curve.bias;
    return 2 * secure_log2_std;
  }
  case NoiseDistribution::TUniform: {
    auto min_bound = curve.slope * size + curve.bias;
    auto log2_modulus = modulus.log2();
    auto log2_modulus_diff = (log2_modulus - 64);
    min_bound = min_bound + log2_modulus_diff;
    min_bound = min_bound.ceil();
    auto secure_log2_var = pow(2., (2. * min_bound) / 3. + 1. / 6.).log2();
    return secure_log2_var - 2 * modulus.log2();
  }
  default:
    break;
  }
  assert(false && "NoiseDistribution not supported");
}

GLWEExpr _minimalVariance(int securityBits, NoiseDistribution noiseDistribution,
                          SecretKeyDistributionKindAttr skDistribution,
                          GLWEExpr size, GLWEExpr modulus) {
  // TODO - Add noise distribution to security curves
  auto curve = ::concretelang::security::getSecurityCurve(
      securityBits,
      securityDistributionToNoiseFormat(skDistribution.getValue()));
  auto log2Modulus = modulus.log2();
  auto epsilonBits = getEpsilonBits(noiseDistribution);
  auto epsilonStdLog2 = epsilonBits - log2Modulus;
  auto epsilonVarLog2 = 2 * epsilonStdLog2;
  auto theoricalSecureVarLog2 =
      unsafeMinimalVarianceLog2(*curve, noiseDistribution, size, modulus);
  return pow(2, max(theoricalSecureVarLog2, epsilonVarLog2));
}

::mlir::Type GLWEType::withMinimalVariance(NoiseDistribution noiseDistribution,
                                           int securityLevel) const {
  if (this->getVariance())
    return *this;

  auto skDistribution = this->getSecretKey().getDistribution().getKind();
  auto size = this->getSecretKey().getSize().getPolySize().getExpr() *
              this->getSecretKey().getSize().getDimension().getExpr();
  auto modulus = this->getEncoding().getBodyModulus().getExpr();
  auto minVar = _minimalVariance(securityLevel, noiseDistribution,
                                 skDistribution, size, modulus);
  return this->withVariance(minVar);
}

::mlir::Type GLWEType::withVariance(GLWEExpr variance) const {
  return GLWEType::get(getContext(), getSecretKey(), getEncoding(),
                       GLWEExprAttr::get(getContext(), variance));
}

::mlir::Type
RadixGLWEType::withMinimalVariance(NoiseDistribution noiseDistribution,
                                   int securityLevel) const {
  if (this->getVariance())
    return *this;

  auto skDistribution = this->getSecretKey().getDistribution().getKind();
  auto size = this->getSecretKey().getSize().getPolySize().getExpr() *
              this->getSecretKey().getSize().getDimension().getExpr();
  auto modulus = this->getEncoding().getBodyModulus().getExpr();
  auto minVar = _minimalVariance(securityLevel, noiseDistribution,
                                 skDistribution, size, modulus);
  return this->withVariance(minVar);
}

::mlir::Type RadixGLWEType::withVariance(GLWEExpr variance) const {
  return RadixGLWEType::get(getContext(), getSecretKey(), getEncoding(),
                            getDecomposition(), getNumberOfNonZerosCoefs(),
                            getMask(), getBody(),
                            GLWEExprAttr::get(getContext(), variance));
}

::mlir::Type GLevType::withMinimalVariance(NoiseDistribution noiseDistribution,
                                           int securityLevel) const {
  if (this->getVariance())
    return *this;

  auto skDistribution = this->getSecretKey().getDistribution().getKind();
  auto size = this->getSecretKey().getSize().getPolySize().getExpr() *
              this->getSecretKey().getSize().getDimension().getExpr();
  auto modulus = this->getEncoding().getBodyModulus().getExpr();
  auto minVar = _minimalVariance(securityLevel, noiseDistribution,
                                 skDistribution, size, modulus);
  return this->withVariance(minVar);
}

::mlir::Type GLevType::withVariance(GLWEExpr variance) const {
  return GLevType::get(getContext(), getSecretKey(), getEncoding(),
                       getDecomposition(), getAverageMessage(), getShape(),
                       GLWEExprAttr::get(getContext(), variance));
}

} // namespace GLWE
} // namespace concretelang
} // namespace mlir