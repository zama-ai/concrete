#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Dialect/GLWE/IR/GLWETypes.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

mlir::LogicalResult ModulusSwitching::verify() {
  auto input = this->getInput().getType().dyn_cast<GLWEType>();
  auto inputParams = input.getParams();
  auto ouput = this->getOutput().getType().dyn_cast<GLWEType>();
  auto outputParams = ouput.getParams();
  // op failed to verify that the input all of GLWE {MessageBound, MaskModulus}
  // parameters matches the output value
  if (inputParams.getMessageBound() != outputParams.getMessageBound() ||
      inputParams.getDimension() != outputParams.getDimension() ||
      inputParams.getPolySize() != outputParams.getPolySize()) {

    this->emitOpError("failed to verify that the input GLWE {message_bound, "
                      "dimension, poly_size} "
                      "parameters are equals to the output value");
    return mlir::failure();
  }
  if (this->getModulus() != outputParams.getMaskModulus()) {
    this->emitOpError("failed to verify that {modulus} parameter is equals to "
                      "the output GLWE {mask_modulus} "
                      "parameter");
    return mlir::failure();
  }
  if (!this->getPartial() &&
      this->getModulus() != outputParams.getBodyModulus()) {
    this->emitOpError("failed to verify that {modulus} parameter is equals to "
                      "the output GLWE {body_modulus} "
                      "parameter");
    return mlir::failure();
  }
  if (this->getPartial() &&
      inputParams.getBodyModulus() != outputParams.getBodyModulus()) {
    this->emitOpError("with {partial = true} failed to verify that the input "
                      "GLWE {body_modulus} "
                      "parameter is equal to the output value");
    return mlir::failure();
  }
  return mlir::success();
}
} // namespace GLWE
} // namespace concretelang
} // namespace mlir