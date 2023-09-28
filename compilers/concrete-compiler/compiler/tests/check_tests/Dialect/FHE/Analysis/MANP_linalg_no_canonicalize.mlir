// RUN: concretecompiler --passes MANP --passes ConcreteOptimizer --action=dump-fhe-no-linalg --split-input-file %s 2>&1 | FileCheck %s

func.func @sum(%0: tensor<5x3x4x2x!FHE.eint<7>>, %35: tensor<2x0x3x!FHE.eint<7>>) -> !FHE.eint<7> {
  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %1 = "FHELinalg.sum"(%0) : (tensor<5x3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %2 = "FHELinalg.sum"(%0) { axes = [0] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %3 = "FHELinalg.sum"(%0) { axes = [1] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %4 = "FHELinalg.sum"(%0) { axes = [2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %5 = "FHELinalg.sum"(%0) { axes = [3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x4x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %6 = "FHELinalg.sum"(%0) { axes = [0, 1] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<4x2x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %7 = "FHELinalg.sum"(%0) { axes = [0, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %8 = "FHELinalg.sum"(%0) { axes = [0, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %9 = "FHELinalg.sum"(%0) { axes = [1, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x2x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %10 = "FHELinalg.sum"(%0) { axes = [1, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x4x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %11 = "FHELinalg.sum"(%0) { axes = [2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x!FHE.eint<7>>

  // CHECK: MANP = 8 : ui{{[0-9]+}}
  %12 = "FHELinalg.sum"(%0) { axes = [0, 1, 2] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>

  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %13 = "FHELinalg.sum"(%0) { axes = [0, 1, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>

  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %14 = "FHELinalg.sum"(%0) { axes = [0, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %15 = "FHELinalg.sum"(%0) { axes = [1, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x!FHE.eint<7>>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %16 = "FHELinalg.sum"(%0) { axes = [0, 1, 2, 3] } : (tensor<5x3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %17 = "FHELinalg.sum"(%0) { keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %18 = "FHELinalg.sum"(%0) { axes = [0], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %19 = "FHELinalg.sum"(%0) { axes = [1], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x4x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %20 = "FHELinalg.sum"(%0) { axes = [2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x1x2x!FHE.eint<7>>

  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %21 = "FHELinalg.sum"(%0) { axes = [3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x4x1x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %22 = "FHELinalg.sum"(%0) { axes = [0, 1], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x4x2x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %23 = "FHELinalg.sum"(%0) { axes = [0, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x1x2x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %24 = "FHELinalg.sum"(%0) { axes = [0, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x4x1x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %25 = "FHELinalg.sum"(%0) { axes = [1, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x1x2x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %26 = "FHELinalg.sum"(%0) { axes = [1, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x4x1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %27 = "FHELinalg.sum"(%0) { axes = [2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x3x1x1x!FHE.eint<7>>

  // CHECK: MANP = 8 : ui{{[0-9]+}}
  %28 = "FHELinalg.sum"(%0) { axes = [0, 1, 2], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x2x!FHE.eint<7>>

  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %29 = "FHELinalg.sum"(%0) { axes = [0, 1, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x4x1x!FHE.eint<7>>

  // CHECK: MANP = 7 : ui{{[0-9]+}}
  %30 = "FHELinalg.sum"(%0) { axes = [0, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x3x1x1x!FHE.eint<7>>

  // CHECK: MANP = 5 : ui{{[0-9]+}}
  %31 = "FHELinalg.sum"(%0) { axes = [1, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<5x1x1x1x!FHE.eint<7>>

  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %32 = "FHELinalg.sum"(%0) { axes = [0, 1, 2, 3], keep_dims = true } : (tensor<5x3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x1x!FHE.eint<7>>

  // ===============================

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %36 = "FHELinalg.sum"(%35) : (tensor<2x0x3x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %37 = "FHELinalg.sum"(%35) { axes = [0] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<0x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %38 = "FHELinalg.sum"(%35) { axes = [1] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %39 = "FHELinalg.sum"(%35) { axes = [2] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x0x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %40 = "FHELinalg.sum"(%35) { axes = [0, 1] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %41 = "FHELinalg.sum"(%35) { axes = [0, 2] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<0x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %42 = "FHELinalg.sum"(%35) { axes = [1, 2] } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %43 = "FHELinalg.sum"(%35) { axes = [0, 1 ,2] } : (tensor<2x0x3x!FHE.eint<7>>) -> !FHE.eint<7>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %44 = "FHELinalg.sum"(%35) { keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %45 = "FHELinalg.sum"(%35) { axes = [0], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x0x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %46 = "FHELinalg.sum"(%35) { axes = [1], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x1x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %47 = "FHELinalg.sum"(%35) { axes = [2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x0x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %48 = "FHELinalg.sum"(%35) { axes = [0, 1], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x1x3x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %49 = "FHELinalg.sum"(%35) { axes = [0, 2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x0x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %50 = "FHELinalg.sum"(%35) { axes = [1, 2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<2x1x1x!FHE.eint<7>>

  // CHECK: MANP = 1 : ui{{[0-9]+}}
  %51 = "FHELinalg.sum"(%35) { axes = [0, 1 ,2], keep_dims = true } : (tensor<2x0x3x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>

  return %1 : !FHE.eint<7>
}

// -----

func.func @concat(%arg0: tensor<4x!FHE.eint<7>>, %arg1:tensor<5x!FHE.eint<7>>, %arg2:tensor<10x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  // CHECK: MANP = 2 : ui{{[0-9]+}}
  %1 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %3 = "FHELinalg.sum"(%arg1) { keep_dims = true } : (tensor<5x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>

  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %5 = "FHELinalg.sum"(%arg2) { keep_dims = true } : (tensor<10x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>

  // CHECK: MANP = 3 : ui{{[0-9]+}}
  %6 = "FHELinalg.concat"(%1, %3) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<2x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %7 = "FHELinalg.concat"(%1, %5) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<2x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %8 = "FHELinalg.concat"(%3, %5) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<2x!FHE.eint<7>>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %9 = "FHELinalg.concat"(%1, %3, %5) : (tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>, tensor<1x!FHE.eint<7>>) ->  tensor<3x!FHE.eint<7>>

  return %9 : tensor<3x!FHE.eint<7>>
}
