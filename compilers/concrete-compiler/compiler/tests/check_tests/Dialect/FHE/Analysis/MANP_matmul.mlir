// RUN: concretecompiler --passes MANP --passes ConcreteOptimizer --action=dump-fhe-no-linalg --split-input-file %s 2>&1 | FileCheck %s

func.func @main(%arg0: tensor<1x10x!FHE.eint<33>>) -> tensor<1x1x!FHE.eint<33>> {
  // sqrt(7282^2 + 20329^2 + 7232^2 + 32768 ^2 + 6446^2 + 32767^2 + 4708^2 + 20050^2 + 28812^2 + 17300^2) = 65277.528491817
  %cst_1 = arith.constant dense<[[-7282], [-20329], [-7232], [-32768], [6446], [32767], [-4708], [-20050], [-28812], [-17300]]> : tensor<10x1xi34>
  // CHECK: MANP = 65278
  %2 = "FHELinalg.matmul_eint_int"(%arg0, %cst_1) : (tensor<1x10x!FHE.eint<33>>, tensor<10x1xi34>) -> tensor<1x1x!FHE.eint<33>>
  return %2 : tensor<1x1x!FHE.eint<33>>
}
