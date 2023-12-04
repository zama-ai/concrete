// RUN: concretecompiler --action=dump-llvm-ir --optimizer-strategy=dag-multi %s
func.func @main(%arg0: tensor<1x1x4x4x!FHE.eint<4>>) -> tensor<1x1x2x2x!FHE.eint<4>> {
  %cst = arith.constant dense<[[[[2, 0], [3, 1]]]]> : tensor<1x1x2x2xi5>
  %cst_0 = arith.constant dense<0> : tensor<1xi5>
  %0 = "FHELinalg.conv2d"(%arg0, %cst, %cst_0) {dilations = dense<1> : tensor<2xi64>, group = 1 : i64, padding = dense<0> : tensor<4xi64>, strides = dense<2> : tensor<2xi64>} : (tensor<1x1x4x4x!FHE.eint<4>>, tensor<1x1x2x2xi5>, tensor<1xi5>) -> tensor<1x1x2x2x!FHE.eint<4>>
  return %0 : tensor<1x1x2x2x!FHE.eint<4>>
}
