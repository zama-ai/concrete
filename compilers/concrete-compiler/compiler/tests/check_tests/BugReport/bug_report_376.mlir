// RUN: concretecompiler --action=dump-llvm-ir --optimizer-strategy=dag-multi %s
// Just ensure that compile
// https://github.com/zama-ai/concrete-internal/issues/376
func.func @main(%arg0: tensor<1x1x2x2x!FHE.eint<7>>) -> tensor<1x1x13x12x!FHE.eint<7>> {
  %c1_i8 = arith.constant 1 : i8
  %0 = "FHE.zero_tensor"() : () -> tensor<1x1x2x7x!FHE.eint<7>>
  %from_elements = tensor.from_elements %c1_i8 : tensor<1xi8>
  %1 = "FHELinalg.add_eint_int"(%0, %from_elements) : (tensor<1x1x2x7x!FHE.eint<7>>, tensor<1xi8>) -> tensor<1x1x2x7x!FHE.eint<7>>
  %c0_i8 = arith.constant 0 : i8
  %from_elements_0 = tensor.from_elements %c0_i8 : tensor<1xi8>
  %2 = "FHELinalg.to_signed"(%1) : (tensor<1x1x2x7x!FHE.eint<7>>) -> tensor<1x1x2x7x!FHE.esint<7>>
  %3 = "FHELinalg.mul_eint_int"(%2, %from_elements_0) : (tensor<1x1x2x7x!FHE.esint<7>>, tensor<1xi8>) -> tensor<1x1x2x7x!FHE.esint<7>>
  %4 = "FHELinalg.to_unsigned"(%3) : (tensor<1x1x2x7x!FHE.esint<7>>) -> tensor<1x1x2x7x!FHE.eint<7>>
  %inserted_slice = tensor.insert_slice %arg0 into %4[0, 0, 0, 3] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x1x2x2x!FHE.eint<7>> into tensor<1x1x2x7x!FHE.eint<7>>
  %5 = "FHE.zero_tensor"() : () -> tensor<1x1x9x10x!FHE.eint<7>>
  %6 = "FHELinalg.add_eint_int"(%5, %from_elements) : (tensor<1x1x9x10x!FHE.eint<7>>, tensor<1xi8>) -> tensor<1x1x9x10x!FHE.eint<7>>
  %7 = "FHELinalg.to_signed"(%6) : (tensor<1x1x9x10x!FHE.eint<7>>) -> tensor<1x1x9x10x!FHE.esint<7>>
  %8 = "FHELinalg.mul_eint_int"(%7, %from_elements_0) : (tensor<1x1x9x10x!FHE.esint<7>>, tensor<1xi8>) -> tensor<1x1x9x10x!FHE.esint<7>>
  %9 = "FHELinalg.to_unsigned"(%8) : (tensor<1x1x9x10x!FHE.esint<7>>) -> tensor<1x1x9x10x!FHE.eint<7>>
  %inserted_slice_1 = tensor.insert_slice %inserted_slice into %9[0, 0, 3, 1] [1, 1, 2, 7] [1, 1, 1, 1] : tensor<1x1x2x7x!FHE.eint<7>> into tensor<1x1x9x10x!FHE.eint<7>>
  %10 = "FHE.zero_tensor"() : () -> tensor<1x1x13x12x!FHE.eint<7>>
  %11 = "FHELinalg.add_eint_int"(%10, %from_elements) : (tensor<1x1x13x12x!FHE.eint<7>>, tensor<1xi8>) -> tensor<1x1x13x12x!FHE.eint<7>>
  %12 = "FHELinalg.to_signed"(%11) : (tensor<1x1x13x12x!FHE.eint<7>>) -> tensor<1x1x13x12x!FHE.esint<7>>
  %13 = "FHELinalg.mul_eint_int"(%12, %from_elements_0) : (tensor<1x1x13x12x!FHE.esint<7>>, tensor<1xi8>) -> tensor<1x1x13x12x!FHE.esint<7>>
  %14 = "FHELinalg.to_unsigned"(%13) : (tensor<1x1x13x12x!FHE.esint<7>>) -> tensor<1x1x13x12x!FHE.eint<7>>
  %inserted_slice_2 = tensor.insert_slice %inserted_slice_1 into %14[0, 0, 2, 1] [1, 1, 9, 10] [1, 1, 1, 1] : tensor<1x1x9x10x!FHE.eint<7>> into tensor<1x1x13x12x!FHE.eint<7>>
  return %inserted_slice_2 : tensor<1x1x13x12x!FHE.eint<7>>
}
