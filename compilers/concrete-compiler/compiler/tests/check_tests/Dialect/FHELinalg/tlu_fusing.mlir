// RUN: concretecompiler --split-input-file --action=dump-fhe --passes canonicalize %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<3>> {
// CHECK-NEXT:   %cst = arith.constant dense<[0, 0, 2, 4]> : tensor<4xi64>
// CHECK-NEXT:   %0 = "FHELinalg.apply_lookup_table"(%arg0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<4xi64>) -> tensor<2x!FHE.eint<3>>
// CHECK-NEXT:   return %0 : tensor<2x!FHE.eint<3>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<3>> {
  %c2_i3 = arith.constant 2 : i3
  %cst = arith.constant dense<[0, 1, 4, 9]> : tensor<4xi64>
  %0 = "FHELinalg.apply_lookup_table"(%arg0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<4xi64>) -> tensor<2x!FHE.eint<4>>
  %cst_0 = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]> : tensor<16xi64>
  %1 = "FHELinalg.apply_lookup_table"(%0, %cst_0) : (tensor<2x!FHE.eint<4>>, tensor<16xi64>) -> tensor<2x!FHE.eint<3>>
  return %1 : tensor<2x!FHE.eint<3>>
}

// -----

// CHECK:      func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<4>> {
// CHECK-NEXT:   %cst = arith.constant dense<{{\[\[0, 0, 2, 4\], \[0, 0, 4, 13\]\]}}> : tensor<2x4xi64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<[0, 1]> : tensor<2xindex>
// CHECK-NEXT:   %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst, %cst_0) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<4>>
// CHECK-NEXT:   return %0 : tensor<2x!FHE.eint<4>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<4>> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xindex>
  %cst_0 = arith.constant dense<[[0, 1, 4, 9], [0, 1, 8, 27]]> : tensor<2x4xi64>
  %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst_0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<5>>
  %c2_i3 = arith.constant 2 : i3
  %cst_1 = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]> : tensor<32xi64>
  %1 = "FHELinalg.apply_lookup_table"(%0, %cst_1) : (tensor<2x!FHE.eint<5>>, tensor<32xi64>) -> tensor<2x!FHE.eint<4>>
  return %1 : tensor<2x!FHE.eint<4>>
}

// -----

// CHECK:      func.func @main(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<4>> {
// CHECK-NEXT:   %cst = arith.constant dense<{{\[\[0, 0, 2, 4\], \[0, 0, 4, 13\]\]}}> : tensor<2x4xi64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<{{\[\[0, 1\], \[1, 0\], \[0, 1\]\]}}> : tensor<3x2xindex>
// CHECK-NEXT:   %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst, %cst_0) : (tensor<3x2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<3x2xindex>) -> tensor<3x2x!FHE.eint<4>>
// CHECK-NEXT:   return %0 : tensor<3x2x!FHE.eint<4>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.eint<4>> {
  %cst = arith.constant dense<[[0, 1], [1, 0], [0, 1]]> : tensor<3x2xindex>
  %cst_0 = arith.constant dense<[[0, 1, 4, 9], [0, 1, 8, 27]]> : tensor<2x4xi64>
  %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst_0, %cst) : (tensor<3x2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<3x2xindex>) -> tensor<3x2x!FHE.eint<5>>
  %c2_i3 = arith.constant 2 : i3
  %cst_1 = arith.constant dense<[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]> : tensor<32xi64>
  %1 = "FHELinalg.apply_lookup_table"(%0, %cst_1) : (tensor<3x2x!FHE.eint<5>>, tensor<32xi64>) -> tensor<3x2x!FHE.eint<4>>
  return %1 : tensor<3x2x!FHE.eint<4>>
}

// -----

// CHECK:      func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<3>> {
// CHECK-NEXT:   %cst = arith.constant dense<{{\[\[0, 0, 2, 4\], \[0, 0, 1, 3\]\]}}> : tensor<2x4xi64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<[0, 1]> : tensor<2xindex>
// CHECK-NEXT:   %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst, %cst_0) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<3>>
// CHECK-NEXT:   return %0 : tensor<2x!FHE.eint<3>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<3>> {
  %c2_i3 = arith.constant 2 : i3
  %cst = arith.constant dense<[0, 1, 4, 9]> : tensor<4xi64>
  %0 = "FHELinalg.apply_lookup_table"(%arg0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<4xi64>) -> tensor<2x!FHE.eint<4>>
  %cst_0 = arith.constant dense<[0, 1]> : tensor<2xindex>
  %cst_1 = arith.constant dense<[[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]]> : tensor<2x16xi64>
  %1 = "FHELinalg.apply_mapped_lookup_table"(%0, %cst_1, %cst_0) : (tensor<2x!FHE.eint<4>>, tensor<2x16xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<3>>
  return %1 : tensor<2x!FHE.eint<3>>
}

// -----

// CHECK:      func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<4>> {
// CHECK-NEXT:   %cst = arith.constant dense<{{\[\[0, 0, 1, 3\], \[0, 0, 4, 13\]\]}}> : tensor<2x4xi64>
// CHECK-NEXT:   %cst_0 = arith.constant dense<[0, 1]> : tensor<2xindex>
// CHECK-NEXT:   %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst, %cst_0) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<4>>
// CHECK-NEXT:   return %0 : tensor<2x!FHE.eint<4>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<4>> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xindex>
  %cst_0 = arith.constant dense<[[0, 1, 4, 9], [0, 1, 8, 27]]> : tensor<2x4xi64>
  %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst_0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<5>>
  %cst_1 = arith.constant dense<[[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]]> : tensor<2x32xi64>
  %1 = "FHELinalg.apply_mapped_lookup_table"(%0, %cst_1, %cst) : (tensor<2x!FHE.eint<5>>, tensor<2x32xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<4>>
  return %1 : tensor<2x!FHE.eint<4>>
}

// -----

// CHECK:      func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<10>> {
// CHECK-NEXT:   %cst = arith.constant dense<[0, 1, 64, 729]> : tensor<4xi64>
// CHECK-NEXT:   %0 = "FHELinalg.apply_lookup_table"(%arg0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<4xi64>) -> tensor<2x!FHE.eint<10>>
// CHECK-NEXT:   return %0 : tensor<2x!FHE.eint<10>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x!FHE.eint<2>>) -> tensor<2x!FHE.eint<10>> {
  %cst = arith.constant dense<[0, 1]> : tensor<2xindex>
  %cst_0 = arith.constant dense<[[0, 1, 4, 9], [0, 1, 8, 27]]> : tensor<2x4xi64>
  %0 = "FHELinalg.apply_mapped_lookup_table"(%arg0, %cst_0, %cst) : (tensor<2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<5>>
  %cst_1 = arith.constant dense<[[0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000, 9261, 10648, 12167, 13824, 15625, 17576, 19683, 21952, 24389, 27000, 29791], [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961]]> : tensor<2x32xi64>
  %1 = "FHELinalg.apply_mapped_lookup_table"(%0, %cst_1, %cst) : (tensor<2x!FHE.eint<5>>, tensor<2x32xi64>, tensor<2xindex>) -> tensor<2x!FHE.eint<10>>
  return %1 : tensor<2x!FHE.eint<10>>
}
