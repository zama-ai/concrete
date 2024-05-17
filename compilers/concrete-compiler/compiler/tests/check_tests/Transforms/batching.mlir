// RUN: concretecompiler --split-input-file --action=dump-batched-tfhe --batch-tfhe-ops --skip-program-info %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @batch_continuous_slice_keyswitch
// CHECK: (%arg0: tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN:.*]]{{\]}}<1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_OUT:.*]]{{\]}}<1,750>>> {
func.func @batch_continuous_slice_keyswitch(%arg0: tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
  // CHECK: %[[V0:.*]] = tensor.collapse_shape [[ARG:.*]] {{\[\[0, 1, 2\]\]}} : tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> into tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
  // CHECK: %[[V1:.*]] = "TFHE.batched_keyswitch_glwe"(%[[V0]]) {key = #TFHE<ksk{{\[}}[[KSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_OUT]]{{\]}}<1,750>, 3, 4>>} : (tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>) -> tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK: %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[\[0, 1, 2\]\]}} : tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>> into tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK: return %[[V2]]

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>) {
	%4 = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
	%5 = "TFHE.keyswitch_glwe"(%4) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>
	%7 = tensor.insert %5 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
	scf.yield %7 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
      }
      scf.yield %3 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
    }
    scf.yield %2 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
  }
  return %1 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
}

// -----

// CHECK-LABEL: func.func @cleanup_continuous_slice
// CHECK: (%arg0: tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK:.*]]{{\]}}<1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK]]{{\]}}<1,2048>>> {
// CHECK: return %arg0 : tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK]]{{\]}}<1,2048>>>
func.func @cleanup_continuous_slice(%arg0: tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
	%4 = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
	%5 = tensor.insert %4 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
	scf.yield %5 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
      }
      scf.yield %3 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
    }
    scf.yield %2 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
  }
  return %1 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
}

// -----

// CHECK-LABEL: func.func @cleanup_continuous_slice_tensor
// CHECK: (%arg0: tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK:.*]]{{\]}}<1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK]]{{\]}}<1,2048>>> {
// CHECK: return %arg0 : tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK]]{{\]}}<1,2048>>>
func.func @cleanup_continuous_slice_tensor(%arg0: tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
	%4 = tensor.extract_slice %arg0[%arg2, %arg4, %arg6] [1, 1, 1] [1, 1, 1] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>> to tensor<1x1x1x!TFHE.glwe<sk<0,1,2048>>>
	%5 = tensor.insert_slice %4 into %arg7[%arg2, %arg4, %arg6][1, 1, 1][1, 1, 1] : tensor<1x1x1x!TFHE.glwe<sk<0,1,2048>>> into tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
	scf.yield %5 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
      }
      scf.yield %3 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
    }
    scf.yield %2 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
  }
  return %1 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
}

// -----

// CHECK-LABEL: func.func @batch_continuous_slice_keyswitch_1dim
// CHECK: (%arg0: tensor<4x!TFHE.glwe<sk{{\[}}[[SK_IN:.*]]{{\]}}<1,2048>>>) -> tensor<4x!TFHE.glwe<sk{{\[}}[[SK_OUT:.*]]{{\]}}<1,750>>> {
func.func @batch_continuous_slice_keyswitch_1dim(%arg0: tensor<4x!TFHE.glwe<sk<0,1,2048>>>) -> tensor<4x!TFHE.glwe<sk<1,1,750>>> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %0 = bufferization.alloc_tensor() : tensor<4x!TFHE.glwe<sk<1,1,750>>>
  // CHECK: %[[V0:.*]] = "TFHE.batched_keyswitch_glwe"(%[[ARG:.*]]) {key = #TFHE<ksk{{\[}}[[KSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_OUT]]{{\]}}<1,750>, 3, 4>>} : (tensor<4x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>) -> tensor<4x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK: return %[[V0]]

  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x!TFHE.glwe<sk<1,1,750>>>) {
    %2 = tensor.extract %arg0[%arg2] : tensor<4x!TFHE.glwe<sk<0,1,2048>>>
    %3 = "TFHE.keyswitch_glwe"(%2) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>
    %4 = tensor.insert %3 into %arg3[%arg2] : tensor<4x!TFHE.glwe<sk<1,1,750>>>
    scf.yield %4 : tensor<4x!TFHE.glwe<sk<1,1,750>>>
  }
  return %1 : tensor<4x!TFHE.glwe<sk<1,1,750>>>
}


// -----

// CHECK-LABEL: func.func @batch_offset_extract_keyswitch
// CHECK: (%arg0: tensor<99x2x3x4x99x99x!TFHE.glwe<sk{{\[}}[[SK_IN:.*]]{{\]}}<1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_OUT:.*]]{{\]}}<1,750>>> {
func.func @batch_offset_extract_keyswitch(%arg0: tensor<99x2x3x4x99x99x!TFHE.glwe<sk<0,1,2048>>>) -> tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c97 = arith.constant 97 : index

  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
  // CHECK: %[[V0:.*]] = tensor.collapse_shape %[[SLICE:.*]] {{\[\[0, 1, 2, 3, 4, 5\]\]}} : tensor<1x2x3x4x1x1x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> into tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
  // CHECK: %[[V1:.*]] = "TFHE.batched_keyswitch_glwe"(%[[V0]]) {key = #TFHE<ksk{{\[}}[[KSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_OUT]]{{\]}}<1,750>, 3, 4>>} : (tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>) -> tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK: %[[V2:.*]] = tensor.expand_shape %[[V1]] {{\[\[0, 1, 2\]\]}} : tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>> into tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK: return %[[V2]]

  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>) {
	%4 = tensor.extract %arg0[%c0, %arg2, %arg4, %arg6, %c97, %c1] : tensor<99x2x3x4x99x99x!TFHE.glwe<sk<0,1,2048>>>
	%5 = "TFHE.keyswitch_glwe"(%4) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>
	%7 = tensor.insert %5 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
	scf.yield %7 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
      }
      scf.yield %3 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
    }
    scf.yield %2 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
  }
  return %1 : tensor<2x3x4x!TFHE.glwe<sk<1,1,750>>>
}

// -----

// CHECK-LABEL: func.func @batch_offset_shifted_bounds_nonunitstep_extract_keyswitch
// CHECK: (%arg0: tensor<99x20x30x40x99x99x!TFHE.glwe<sk{{\[}}[[SK_IN:.*]]{{\]}}<1,2048>>>) -> tensor<2x2x2x!TFHE.glwe<sk{{\[}}[[SK_OUT:.*]]{{\]}}<1,750>>> {
func.func @batch_offset_shifted_bounds_nonunitstep_extract_keyswitch(%arg0: tensor<99x20x30x40x99x99x!TFHE.glwe<sk<0,1,2048>>>) -> tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %c97 = arith.constant 97 : index
  %c23 = arith.constant 23 : index

  %0 = bufferization.alloc_tensor() : tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>

  // CHECK: %[[V1:.*]] = tensor.extract_slice %arg0{{\[0, 3, 7, 9, 97, 1\] \[1, 2, 2, 2, 1, 1\] \[1, 2, 1, 7, 1, 1\]}} : tensor<99x20x30x40x99x99x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> to tensor<1x2x2x2x1x1x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
  // CHECK-NEXT: %[[V3:.*]] = tensor.collapse_shape %[[V1]] {{\[\[0, 1, 2, 3, 4, 5\]\]}} : tensor<1x2x2x2x1x1x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> into tensor<8x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
  // CHECK-NEXT: %[[V4:.*]] = "TFHE.batched_keyswitch_glwe"(%[[V3]]) {key = #TFHE<ksk{{\[}}[[KSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_OUT]]{{\]}}<1,750>, 3, 4>>} : (tensor<8x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>) -> tensor<8x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK-NEXT: %[[V5:.*]] = tensor.expand_shape %[[V4]] {{\[\[0, 1, 2\]\]}} : tensor<8x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>> into tensor<2x2x2x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
  // CHECK-NEXT: return %[[V5]] : tensor<2x2x2x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>

  %1 = scf.for %arg2 = %c3 to %c7 step %c2 iter_args(%arg3 = %0) -> (tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>) {
    %2 = scf.for %arg4 = %c7 to %c9 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>) {
      %3 = scf.for %arg6 = %c9 to %c23 step %c7 iter_args(%arg7 = %arg5) -> (tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>) {
	%4 = tensor.extract %arg0[%c0, %arg2, %arg4, %arg6, %c97, %c1] : tensor<99x20x30x40x99x99x!TFHE.glwe<sk<0,1,2048>>>
	%5 = "TFHE.keyswitch_glwe"(%4) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>

	%od00 = arith.subi %arg2, %c3 : index
	%od01 = arith.divsi %od00, %c2 : index

	%od10 = arith.subi %arg4, %c7 : index

	%od20 = arith.subi %arg6, %c9 : index
	%od21 = arith.divsi %od20, %c7 : index

	%7 = tensor.insert %5 into %arg7[%od01, %od10, %od21] : tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>
	scf.yield %7 : tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>
      }
      scf.yield %3 : tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>
    }
    scf.yield %2 : tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>
  }
  return %1 : tensor<2x2x2x!TFHE.glwe<sk<1,1,750>>>
}

// -----

// CHECK-LABEL: func.func @apply_lookup_table_contiguous
// CHECK: (%arg0: tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN:.*]]{{\]}}<1,2048>>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> {
// CHECK:   %[[Vcollapsed:.*]] = tensor.collapse_shape %[[Varg0:.*]] {{\[\[0, 1, 2\]\]}} : tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> into tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
// CHECK-NEXT:   %[[V1:.*]] = "TFHE.batched_keyswitch_glwe"(%[[Vcollapsed]]) {key = #TFHE<ksk{{\[}}[[KSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_OUT:.*]]{{\]}}<1,750>, 3, 4>>} : (tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>) -> tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
// CHECK-NEXT:   %[[V2:.*]] = "TFHE.encode_expand_lut_for_bootstrap"(%[[Varg1:.*]]) {isSigned = false, outputBits = 2 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<1024xi64>
// CHECK-NEXT:   %[[V3:.*]] = "TFHE.batched_bootstrap_glwe"(%[[V1]], %[[V2]]) {key = #TFHE<bsk{{\[}}[[BSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, 1024, 2, 1, 23>>} : (tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>, tensor<1024xi64>) -> tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
// CHECK-NEXT:   %[[Vexpanded:.*]] = tensor.expand_shape %[[V3]] {{\[\[0, 1, 2\]\]}} : tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> into tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>
func.func @apply_lookup_table_contiguous(%arg0: tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
        %extracted = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
        %4 = "TFHE.encode_expand_lut_for_bootstrap"(%arg1) {isSigned = false, outputBits = 2 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<1024xi64>
        %5 = "TFHE.keyswitch_glwe"(%extracted) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>
        %6 = "TFHE.bootstrap_glwe"(%5, %4) {key = #TFHE.bsk<sk<0,1,2048>, sk<0,1,2048>, 1024, 2, 1, 23>} : (!TFHE.glwe<sk<1,1,750>>, tensor<1024xi64>) -> !TFHE.glwe<sk<0,1,2048>>
        %inserted = tensor.insert %6 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
        scf.yield %inserted : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
      }
      scf.yield %3 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
    }
    scf.yield %2 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
  }
  return %1 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
}

// -----

// CHECK-LABEL: func.func @apply_lookup_table_contiguous_hoistable_nonbatchable
// CHECK: (%arg0: tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN:.*]]{{\]}}<1,2048>>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>> {
// CHECK:   %[[V1:.*]] = "TFHE.batched_keyswitch_glwe"(%[[Vcollapsed]]) {key = #TFHE<ksk{{\[}}[[KSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_OUT:.*]]{{\]}}<1,750>, 3, 4>>} : (tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>) -> tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>
// CHECK:   %[[V3:.*]] = "TFHE.batched_bootstrap_glwe"(%[[V1]], %[[V2:.*]]) {key = #TFHE<bsk{{\[}}[[BSK:.*]]{{\]}}<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, sk{{\[}}[[SK_IN]]{{\]}}<1,2048>, 1024, 2, 1, 23>>} : (tensor<24x!TFHE.glwe<sk{{\[}}[[SK_OUT]]{{\]}}<1,750>>>, tensor<1024xi64>) -> tensor<24x!TFHE.glwe<sk{{\[}}[[SK_IN]]{{\]}}<1,2048>>>
func.func @apply_lookup_table_contiguous_hoistable_nonbatchable(%arg0: tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %0 = bufferization.alloc_tensor() : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
  %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
    %2 = scf.for %arg4 = %c0 to %c3 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
      %3 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %arg5) -> (tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>) {
        %extracted = tensor.extract %arg0[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
        %4 = "TFHE.encode_expand_lut_for_bootstrap"(%arg1) {isSigned = false, outputBits = 2 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<1024xi64>
	%aa = tensor.from_elements
	   %c1_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c1_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c1_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c1_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64,
	   %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64, %c0_i64
	   : tensor<1024xi64>
	%b = tensor.extract_slice %4[0][255][1] : tensor<1024xi64> to tensor<255xi64>
	%x = tensor.extract %4[%c0] : tensor<1024xi64>
	%c = tensor.insert_slice %b into %aa[1][255][1] : tensor<255xi64> into tensor<1024xi64>
	%d = tensor.insert %x into %c[%c0] : tensor<1024xi64>
        %5 = "TFHE.keyswitch_glwe"(%extracted) {key = #TFHE.ksk<sk<0,1,2048>, sk<1,1,750>, 3, 4>} : (!TFHE.glwe<sk<0,1,2048>>) -> !TFHE.glwe<sk<1,1,750>>
        %6 = "TFHE.bootstrap_glwe"(%5, %d) {key = #TFHE.bsk<sk<0,1,2048>, sk<0,1,2048>, 1024, 2, 1, 23>} : (!TFHE.glwe<sk<1,1,750>>, tensor<1024xi64>) -> !TFHE.glwe<sk<0,1,2048>>
        %inserted = tensor.insert %6 into %arg7[%arg2, %arg4, %arg6] : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
        scf.yield %inserted : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
      }
      scf.yield %3 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
    }
    scf.yield %2 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
  }
  return %1 : tensor<2x3x4x!TFHE.glwe<sk<0,1,2048>>>
}
