// RUN: concretecompiler --split-input-file --action=dump-concrete-with-loops --batch-concrete-ops %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @batch_continuous_slice_keyswitch(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> {
func.func @batch_continuous_slice_keyswitch(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V0:.*]] = tensor.collapse_shape [[ARG:.*]] {{\[\[0, 1, 2\]\]}} : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> into tensor<24x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V1:.*]] = "Concrete.batched_keyswitch_lwe"(%[[V0]]) {baseLog = 2 : i32, level = 5 : i32} : (tensor<24x!Concrete.lwe_ciphertext<572,2>>) -> tensor<24x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[\[0, 1, 2\]\]}} : tensor<24x!Concrete.lwe_ciphertext<572,2>> into tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: return %[[V2]]

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) {
	%4 = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
	%5 = "Concrete.keyswitch_lwe"(%4) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<572,2>) -> !Concrete.lwe_ciphertext<572,2>
	%7 = tensor.insert %5 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
	scf.yield %7 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
      }
      scf.yield %3 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
    }
    scf.yield %2 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  }
  return %1 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
}

// -----

// CHECK-LABEL: func.func @batch_continuous_slice_keyswitch_1dim(%arg0: tensor<4x!Concrete.lwe_ciphertext<572,2>>) -> tensor<4x!Concrete.lwe_ciphertext<572,2>> {
func.func @batch_continuous_slice_keyswitch_1dim(%arg0: tensor<4x!Concrete.lwe_ciphertext<572,2>>) -> tensor<4x!Concrete.lwe_ciphertext<572,2>> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V0:.*]] = "Concrete.batched_keyswitch_lwe"(%[[ARG:.*]]) {baseLog = 2 : i32, level = 5 : i32} : (tensor<4x!Concrete.lwe_ciphertext<572,2>>) -> tensor<4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: return %[[V0]]

  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x!Concrete.lwe_ciphertext<572,2>>) {
    %2 = tensor.extract %arg0[%arg2] : tensor<4x!Concrete.lwe_ciphertext<572,2>>
    %3 = "Concrete.keyswitch_lwe"(%2) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<572,2>) -> !Concrete.lwe_ciphertext<572,2>
    %4 = tensor.insert %3 into %arg3[%arg2] : tensor<4x!Concrete.lwe_ciphertext<572,2>>
    scf.yield %4 : tensor<4x!Concrete.lwe_ciphertext<572,2>>
  }
  return %1 : tensor<4x!Concrete.lwe_ciphertext<572,2>>
}

// -----

// CHECK-LABEL: func.func @batch_continuous_slice_bootstrap(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>> {
func.func @batch_continuous_slice_bootstrap(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>

  // CHECK: %[[V0:.*]] = tensor.collapse_shape [[ARG:.*]] {{\[\[0, 1, 2\]\]}} : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> into tensor<24x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V1:.*]] = "Concrete.batched_bootstrap_lwe"(%[[V0]], %arg1) {baseLog = 8 : i32, glweDimension = 1 : i32, level = 2 : i32, polySize = 1024 : i32} : (tensor<24x!Concrete.lwe_ciphertext<572,2>>, tensor<4xi64>) -> tensor<24x!Concrete.lwe_ciphertext<1024,2>>
  // CHECK: %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[\[0, 1, 2\]\]}} : tensor<24x!Concrete.lwe_ciphertext<1024,2>> into tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
  // CHECK: return %[[V2]]

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>) {
	%4 = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
	%5 = "Concrete.bootstrap_lwe"(%4, %arg1) {baseLog = 8 : i32, glweDimension = 1 : i32, level = 2 : i32, polySize = 1024 : i32} : (!Concrete.lwe_ciphertext<572,2>, tensor<4xi64>) -> !Concrete.lwe_ciphertext<1024,2>
	%6 = tensor.insert %5 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
	scf.yield %6 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
      }
      scf.yield %3 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
    }
    scf.yield %2 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
  }
  return %1 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
}

// -----

// CHECK-LABEL: func.func @batch_continuous_slice_apply_lookup_table(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>> {
func.func @batch_continuous_slice_apply_lookup_table(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V0:.*]] = tensor.collapse_shape [[ARG:.*]] {{\[\[0, 1, 2\]\]}} : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>> into tensor<24x!Concrete.lwe_ciphertext<1024,2>>
  // CHECK: %[[V1:.*]] = "Concrete.batched_keyswitch_lwe"(%[[V0]]) {baseLog = 2 : i32, level = 5 : i32} : (tensor<24x!Concrete.lwe_ciphertext<1024,2>>) -> tensor<24x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V2:.*]] = "Concrete.batched_bootstrap_lwe"(%[[V1]], %arg1) {baseLog = 8 : i32, glweDimension = 1 : i32, level = 2 : i32, polySize = 1024 : i32} : (tensor<24x!Concrete.lwe_ciphertext<572,2>>, tensor<4xi64>) -> tensor<24x!Concrete.lwe_ciphertext<1024,2>>
  // CHECK: %[[V3:.*]] = tensor.expand_shape %[[V2]] {{\[\[0, 1, 2\]\]}} : tensor<24x!Concrete.lwe_ciphertext<1024,2>> into tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
  // CHECK: return %[[V3]]

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>) {
	%4 = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
	%5 = "Concrete.keyswitch_lwe"(%4) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<1024,2>) -> !Concrete.lwe_ciphertext<572,2>
	%6 = "Concrete.bootstrap_lwe"(%5, %arg1) {baseLog = 8 : i32, glweDimension = 1 : i32, level = 2 : i32, polySize = 1024 : i32} : (!Concrete.lwe_ciphertext<572,2>, tensor<4xi64>) -> !Concrete.lwe_ciphertext<1024,2>
	%7 = tensor.insert %6 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
	scf.yield %7 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
      }
      scf.yield %3 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
    }
    scf.yield %2 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
  }
  return %1 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,2>>
}

// -----

// CHECK-LABEL: func.func @batch_offset_extract_keyswitch(%arg0: tensor<99x2x3x4x99x99x!Concrete.lwe_ciphertext<572,2>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> {
func.func @batch_offset_extract_keyswitch(%arg0: tensor<99x2x3x4x99x99x!Concrete.lwe_ciphertext<572,2>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c97 = arith.constant 97 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[VDROP1DIMS:.*]] = tensor.collapse_shape [[ARG:.*]] {{\[\[0, 1\], \[2\], \[3, 4, 5\]\]}} : tensor<1x2x3x4x1x1x!Concrete.lwe_ciphertext<572,2>> into tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V0:.*]] = tensor.collapse_shape %[[VDROP1DIMS]] {{\[\[0, 1, 2\]\]}} : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>> into tensor<24x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V1:.*]] = "Concrete.batched_keyswitch_lwe"(%[[V0]]) {baseLog = 2 : i32, level = 5 : i32} : (tensor<24x!Concrete.lwe_ciphertext<572,2>>) -> tensor<24x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[\[0, 1, 2\]\]}} : tensor<24x!Concrete.lwe_ciphertext<572,2>> into tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  // CHECK: return %[[V2]]

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>) {
	%4 = tensor.extract %arg0[%c0, %arg2, %arg4, %arg6, %c97, %c1] : tensor<99x2x3x4x99x99x!Concrete.lwe_ciphertext<572,2>>
	%5 = "Concrete.keyswitch_lwe"(%4) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<572,2>) -> !Concrete.lwe_ciphertext<572,2>
	%7 = tensor.insert %5 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
	scf.yield %7 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
      }
      scf.yield %3 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
    }
    scf.yield %2 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
  }
  return %1 : tensor<2x3x4x!Concrete.lwe_ciphertext<572,2>>
}

// -----

// CHECK-LABEL: func.func @batch_offset_shifted_bounds_nonunitstep_extract_keyswitch(%arg0: tensor<99x20x30x40x99x99x!Concrete.lwe_ciphertext<572,2>>) -> tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>> {
func.func @batch_offset_shifted_bounds_nonunitstep_extract_keyswitch(%arg0: tensor<99x20x30x40x99x99x!Concrete.lwe_ciphertext<572,2>>) -> tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %c97 = arith.constant 97 : index
  %c24 = arith.constant 24 : index

  %0 = bufferization.alloc_tensor() : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>

  // CHECK:      %[[V0:.*]] = bufferization.alloc_tensor() : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
  // CHECK-NEXT: %[[V1:.*]] = tensor.extract_slice %arg0{{\[0, 3, 7, 9, 97, 1\] \[1, 2, 2, 2, 1, 1\] \[1, 2, 1, 7, 1, 1\]}} : tensor<99x20x30x40x99x99x!Concrete.lwe_ciphertext<572,2>> to tensor<1x2x2x2x1x1x!Concrete.lwe_ciphertext<572,2>>
  // CHECK-NEXT: %[[V2:.*]] = tensor.collapse_shape %[[V1]] {{\[\[0, 1\], \[2\], \[3, 4, 5\]\]}} : tensor<1x2x2x2x1x1x!Concrete.lwe_ciphertext<572,2>> into tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
  // CHECK-NEXT: %[[V3:.*]] = tensor.collapse_shape %[[V2]] {{\[\[0, 1, 2\]\]}} : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>> into tensor<8x!Concrete.lwe_ciphertext<572,2>>
  // CHECK-NEXT: %[[V4:.*]] = "Concrete.batched_keyswitch_lwe"(%[[V3]]) {baseLog = 2 : i32, level = 5 : i32} : (tensor<8x!Concrete.lwe_ciphertext<572,2>>) -> tensor<8x!Concrete.lwe_ciphertext<572,2>>
  // CHECK-NEXT: %[[V5:.*]] = tensor.expand_shape %[[V4]] {{\[\[0, 1, 2\]\]}} : tensor<8x!Concrete.lwe_ciphertext<572,2>> into tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
  // CHECK-NEXT: return %[[V5]] : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>

  %1 = scf.for %arg2 = %c3 to %c7 step %c2 iter_args(%arg3 = %0) -> (tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>) {
    %2 = scf.for %arg4 = %c7 to %c9 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>) {
      %3 = scf.for %arg6 = %c9 to %c24 step %c7 iter_args(%arg7 = %arg5) -> (tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>) {
	%4 = tensor.extract %arg0[%c0, %arg2, %arg4, %arg6, %c97, %c1] : tensor<99x20x30x40x99x99x!Concrete.lwe_ciphertext<572,2>>
	%5 = "Concrete.keyswitch_lwe"(%4) {baseLog = 2 : i32, level = 5 : i32} : (!Concrete.lwe_ciphertext<572,2>) -> !Concrete.lwe_ciphertext<572,2>

	%od00 = arith.subi %arg2, %c3 : index
	%od01 = arith.divsi %od00, %c2 : index

	%od10 = arith.subi %arg4, %c7 : index

	%od20 = arith.subi %arg6, %c9 : index
	%od21 = arith.divsi %od20, %c7 : index

	%7 = tensor.insert %5 into %arg7[%od01, %od10, %od21] : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
	scf.yield %7 : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
      }
      scf.yield %3 : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
    }
    scf.yield %2 : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
  }
  return %1 : tensor<2x2x2x!Concrete.lwe_ciphertext<572,2>>
}
