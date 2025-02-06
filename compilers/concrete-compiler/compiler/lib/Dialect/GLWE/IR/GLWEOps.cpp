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
} // namespace GLWE
} // namespace concretelang
} // namespace mlir