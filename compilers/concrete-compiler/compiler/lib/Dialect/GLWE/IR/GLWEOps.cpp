#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

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
    this->emitOpError(
        "failed to verify that the glev size is equals to the input secret key "
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
    this->emitOpError(
        "failed to verify that the input and glev {encoding.body_modulus} are "
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
        "{mask_modulus, body_modulus} parameter equal to the {mask_modulus} of "
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
} // namespace GLWE
} // namespace concretelang
} // namespace mlir