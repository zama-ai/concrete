// RUN: concretecompiler --action=dump-tfhe --force-encoding crt %s
func.func @main(%2: tensor<1x1x!FHE.eint<16>>) -> tensor<1x1x1x!FHE.eint<16>> {
  %3 = tensor.expand_shape %2 [[0], [1, 2]] : tensor<1x1x!FHE.eint<16>> into tensor<1x1x1x!FHE.eint<16>>
  return %3 : tensor<1x1x1x!FHE.eint<16>>
}

func.func @main2(%2: tensor<1x1x1x!FHE.eint<16>>) -> tensor<1x1x!FHE.eint<16>> {
  %3 = tensor.collapse_shape %2 [[0], [1, 2]] : tensor<1x1x1x!FHE.eint<16>> into tensor<1x1x!FHE.eint<16>>
  return %3 : tensor<1x1x!FHE.eint<16>>
}


