// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHELinalg.to_signed' op input and output tensors should have the same width
func.func @bad_result_width(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<3>> {
  %1 = "FHELinalg.to_signed"(%arg0): (tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<3>>
  return %1: tensor<3x2x!FHE.esint<3>>
}

// -----

// CHECK-LABEL: error: 'FHELinalg.to_signed' op input and output tensors should have the same shape
func.func @bad_result_shape(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x!FHE.esint<2>> {
  %1 = "FHELinalg.to_signed"(%arg0): (tensor<3x2x!FHE.eint<2>>) -> tensor<3x!FHE.esint<2>>
  return %1: tensor<3x!FHE.esint<2>>
}
