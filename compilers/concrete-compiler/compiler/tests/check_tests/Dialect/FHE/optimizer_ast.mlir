// RUN: concretecompiler --verbose --passes canonicalize --passes MANP --passes ConcreteOptimizer --split-input-file --action=dump-fhe-no-linalg  %s 2>&1| FileCheck %s

func.func @main(%arg0: tensor<5x!FHE.eint<5>>) -> !FHE.eint<5> {
  %weights = arith.constant dense<[-1, -1, -1, -1, -1]> : tensor<5xi6>
  %tlu = arith.constant dense<[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<32xi64>
  %0 = "FHELinalg.apply_lookup_table"(%arg0, %tlu) : (tensor<5x!FHE.eint<5>>, tensor<32xi64>) -> tensor<5x!FHE.eint<5>>
  // CHECK: Dot { [[a:.*]], weights: ClearTensor { shape: Shape { dimensions_size: [5] }, values: [-1, -1, -1, -1, -1] }, kind: Tensor }
  %1 = "FHELinalg.dot_eint_int"(%0, %weights) : (tensor<5x!FHE.eint<5>>, tensor<5xi6>) -> !FHE.eint<5>
  return %1 : !FHE.eint<5>
}
