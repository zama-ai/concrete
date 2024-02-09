// RUN: concretecompiler --action=dump-tfhe --optimizer-strategy=V0 --force-encoding crt --skip-program-info %s
func.func @main(%arg0: tensor<32x!FHE.eint<8>>) -> tensor<16x!FHE.eint<8>>{
 %0 = tensor.extract_slice %arg0[16] [16] [1] : tensor<32x!FHE.eint<8>> to tensor<16x!FHE.eint<8>>
 return %0 : tensor<16x!FHE.eint<8>>
}

func.func @main2(%t0: tensor<2x10x!FHE.eint<6>>, %t1: tensor<2x2x!FHE.eint<6>>) -> tensor<2x10x!FHE.eint<6>> {
  %r = tensor.insert_slice %t1 into %t0[0, 5][2, 2][1, 1] : tensor<2x2x!FHE.eint<6>> into tensor<2x10x!FHE.eint<6>>
  return %r : tensor<2x10x!FHE.eint<6>>
}
