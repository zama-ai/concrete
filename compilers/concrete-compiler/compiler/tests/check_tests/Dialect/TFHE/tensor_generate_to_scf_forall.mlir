// RUN: concretecompiler --split-input-file --action=dump-tfhe %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @from_1d_to_1d(%arg0: tensor<5x!TFHE.glwe<sk?>> {TFHE.OId = 0 : i32}, %arg1: tensor<3xindex>) -> tensor<3x!TFHE.glwe<sk?>> {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT:   %1 = scf.forall (%arg2) in (3) shared_outs(%arg3 = %0) -> (tensor<3x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:     %extracted = tensor.extract %arg1[%arg2] : tensor<3xindex>
// CHECK-NEXT:     %extracted_0 = tensor.extract %arg0[%extracted] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:     %from_elements = tensor.from_elements %extracted_0 : tensor<1x!TFHE.glwe<sk?>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %from_elements into %arg3[%arg2] [1] [1] : tensor<1x!TFHE.glwe<sk?>> into tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1 : tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT: }
func.func @from_1d_to_1d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_to_2d(%arg0: tensor<5x!TFHE.glwe<sk?>> {TFHE.OId = 0 : i32}, %arg1: tensor<2x3xindex>) -> tensor<2x3x!TFHE.glwe<sk?>> {
// CHECK-NEXT:   %0 = tensor.empty() : tensor<2x3x!TFHE.glwe<sk?>>
// CHECK-NEXT:   %1 = scf.forall (%arg2, %arg3) in (2, 3) shared_outs(%arg4 = %0) -> (tensor<2x3x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:     %extracted = tensor.extract %arg1[%arg2, %arg3] : tensor<2x3xindex>
// CHECK-NEXT:     %extracted_0 = tensor.extract %arg0[%extracted] : tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT:     %from_elements = tensor.from_elements %extracted_0 : tensor<1x!TFHE.glwe<sk?>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %from_elements into %arg4[%arg2, %arg3] [1, 1] [1, 1] : tensor<1x!TFHE.glwe<sk?>> into tensor<2x3x!TFHE.glwe<sk?>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1 : tensor<2x3x!TFHE.glwe<sk?>>
// CHECK-NEXT: }
func.func @from_1d_to_2d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_1d(%arg0: tensor<4x5x!TFHE.glwe<sk?>> {TFHE.OId = 0 : i32}, %arg1: tensor<3x2xindex>) -> tensor<3x!TFHE.glwe<sk?>> {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = tensor.empty() : tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT:   %1 = scf.forall (%arg2) in (3) shared_outs(%arg3 = %0) -> (tensor<3x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:     %extracted = tensor.extract %arg1[%arg2, %c0] : tensor<3x2xindex>
// CHECK-NEXT:     %extracted_0 = tensor.extract %arg1[%arg2, %c1] : tensor<3x2xindex>
// CHECK-NEXT:     %extracted_1 = tensor.extract %arg0[%extracted, %extracted_0] : tensor<4x5x!TFHE.glwe<sk?>>
// CHECK-NEXT:     %from_elements = tensor.from_elements %extracted_1 : tensor<1x!TFHE.glwe<sk?>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %from_elements into %arg3[%arg2] [1] [1] : tensor<1x!TFHE.glwe<sk?>> into tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1 : tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT: }
func.func @from_2d_to_1d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_2d(%arg0: tensor<4x5x!TFHE.glwe<sk?>> {TFHE.OId = 0 : i32}, %arg1: tensor<2x3x2xindex>) -> tensor<2x3x!TFHE.glwe<sk?>> {
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = tensor.empty() : tensor<2x3x!TFHE.glwe<sk?>>
// CHECK-NEXT:   %1 = scf.forall (%arg2, %arg3) in (2, 3) shared_outs(%arg4 = %0) -> (tensor<2x3x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:     %extracted = tensor.extract %arg1[%arg2, %arg3, %c0] : tensor<2x3x2xindex>
// CHECK-NEXT:     %extracted_0 = tensor.extract %arg1[%arg2, %arg3, %c1] : tensor<2x3x2xindex>
// CHECK-NEXT:     %extracted_1 = tensor.extract %arg0[%extracted, %extracted_0] : tensor<4x5x!TFHE.glwe<sk?>>
// CHECK-NEXT:     %from_elements = tensor.from_elements %extracted_1 : tensor<1x!TFHE.glwe<sk?>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %from_elements into %arg4[%arg2, %arg3] [1, 1] [1, 1] : tensor<1x!TFHE.glwe<sk?>> into tensor<2x3x!TFHE.glwe<sk?>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1 : tensor<2x3x!TFHE.glwe<sk?>>
// CHECK-NEXT: }
func.func @from_2d_to_2d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_1d(%arg0: tensor<2x4x5x!TFHE.glwe<sk?>> {TFHE.OId = 0 : i32}, %arg1: tensor<3x3xindex>) -> tensor<3x!TFHE.glwe<sk?>> {
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = tensor.empty() : tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT:   %1 = scf.forall (%arg2) in (3) shared_outs(%arg3 = %0) -> (tensor<3x!TFHE.glwe<sk?>>) {
// CHECK-NEXT:     %extracted = tensor.extract %arg1[%arg2, %c0] : tensor<3x3xindex>
// CHECK-NEXT:     %extracted_0 = tensor.extract %arg1[%arg2, %c1] : tensor<3x3xindex>
// CHECK-NEXT:     %extracted_1 = tensor.extract %arg1[%arg2, %c2] : tensor<3x3xindex>
// CHECK-NEXT:     %extracted_2 = tensor.extract %arg0[%extracted, %extracted_0, %extracted_1] : tensor<2x4x5x!TFHE.glwe<sk?>>
// CHECK-NEXT:     %from_elements = tensor.from_elements %extracted_2 : tensor<1x!TFHE.glwe<sk?>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %from_elements into %arg3[%arg2] [1] [1] : tensor<1x!TFHE.glwe<sk?>> into tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %1 : tensor<3x!TFHE.glwe<sk?>>
// CHECK-NEXT: }
func.func @from_3d_to_1d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}
