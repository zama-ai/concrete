// RUN: concretecompiler --action=dump-llvm-ir --optimizer-strategy=dag-multi %s
// Just ensure that compile
// https://github.com/zama-ai/concrete-internal/issues/352
  func.func @main(%arg0: tensor<1x1x8x8x!FHE.esint<2>>, %arg1: tensor<1x1x8x8x!FHE.esint<2>>) -> tensor<1x1x8x8x!FHE.esint<2>> {
    %cst = arith.constant dense<1> : tensor<1x1x1x1xi3>
    %0 = "FHELinalg.conv2d"(%arg0, %cst) {dilations = dense<1> : tensor<2xi64>, group = 1 : i64, padding = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<1x1x1x1xi3>) -> tensor<1x1x8x8x!FHE.esint<2>>
    %1 = "FHELinalg.conv2d"(%arg1, %cst) {dilations = dense<1> : tensor<2xi64>, group = 1 : i64, padding = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<1x1x1x1xi3>) -> tensor<1x1x8x8x!FHE.esint<2>>
    %cst_0 = arith.constant dense<[0, 1, -2, -1]> : tensor<4xi64>
    %2 = "FHELinalg.apply_lookup_table"(%0, %cst_0) : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<4xi64>) -> tensor<1x1x8x8x!FHE.esint<3>>
    %cst_1 = arith.constant dense<[-2, -2, 0, -1]> : tensor<4xi64>
    %3 = "FHELinalg.apply_lookup_table"(%1, %cst_1) : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<4xi64>) -> tensor<1x1x8x8x!FHE.esint<2>>
    %c1_i4 = arith.constant 1 : i4
    %from_elements = tensor.from_elements %c1_i4 : tensor<1xi4>
    %4 = "FHELinalg.add_eint_int"(%2, %from_elements) : (tensor<1x1x8x8x!FHE.esint<3>>, tensor<1xi4>) -> tensor<1x1x8x8x!FHE.esint<3>>
    %cst_2 = arith.constant dense<[0, 4, 7, 11, -14, -11, -7, -4]> : tensor<8xi64>
    %5 = "FHELinalg.apply_lookup_table"(%4, %cst_2) : (tensor<1x1x8x8x!FHE.esint<3>>, tensor<8xi64>) -> tensor<1x1x8x8x!FHE.esint<5>>
    %c1_i3 = arith.constant 1 : i3
    %from_elements_3 = tensor.from_elements %c1_i3 : tensor<1xi3>
    %6 = "FHELinalg.add_eint_int"(%3, %from_elements_3) : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<1xi3>) -> tensor<1x1x8x8x!FHE.esint<2>>
    %cst_4 = arith.constant dense<[0, 9, -19, -9]> : tensor<4xi64>
    %7 = "FHELinalg.apply_lookup_table"(%6, %cst_4) : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<4xi64>) -> tensor<1x1x8x8x!FHE.esint<5>>
    %8 = "FHELinalg.add_eint"(%5, %7) : (tensor<1x1x8x8x!FHE.esint<5>>, tensor<1x1x8x8x!FHE.esint<5>>) -> tensor<1x1x8x8x!FHE.esint<5>>
    %cst_5 = arith.constant dense<[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0]> : tensor<32xi64>
    %9 = "FHELinalg.apply_lookup_table"(%8, %cst_5) : (tensor<1x1x8x8x!FHE.esint<5>>, tensor<32xi64>) -> tensor<1x1x8x8x!FHE.esint<2>>
    %10 = "FHELinalg.conv2d"(%9, %cst) {dilations = dense<1> : tensor<2xi64>, group = 1 : i64, padding = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x1x8x8x!FHE.esint<2>>, tensor<1x1x1x1xi3>) -> tensor<1x1x8x8x!FHE.esint<2>>
    return %10 : tensor<1x1x8x8x!FHE.esint<2>>
  }
