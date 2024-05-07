// RUN: concretecompiler --split-input-file --action=dump-parametrized-tfhe --optimizer-strategy=dag-multi --skip-program-info %s 2>&1| FileCheck %s

// CHECK:      func.func @funconly_fwd(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:    return %arg0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:  }
func.func @funconly_fwd(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk?> {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  return %a0: !TFHE.glwe<sk?>
}

// -----

// CHECK:      func.func @funconly_bwd(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:    return %arg0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:  }
func.func @funconly_bwd(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk[1]<12,1024>> {
  %a0 = "TypeInference.propagate_upward"(%arg0) : (!TFHE.glwe<sk?>) -> (!TFHE.glwe<sk[1]<12,1024>>)
  return %a0: !TFHE.glwe<sk[1]<12,1024>>
}

// -----

// CHECK:       func.func @funconly_fwd_multires(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[2]<13,2048>>) -> (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[2]<13,2048>>) {
// CHECK-NEXT:    return %arg0, %arg1 : !TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[2]<13,2048>>
// CHECK-NEXT:  }
func.func @funconly_fwd_multires(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[2]<13,2048>>) -> (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a1 = "TypeInference.propagate_downward"(%arg1) : (!TFHE.glwe<sk[2]<13,2048>>) -> (!TFHE.glwe<sk?>)
  return %a0, %a1: !TFHE.glwe<sk?>, !TFHE.glwe<sk?>
}

// -----

// CHECK:      func.func @fwd1(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:    %0 = "TFHE.add_glwe"(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:    return %0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:  }
func.func @fwd1(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk?> {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a1 = "TypeInference.propagate_downward"(%arg1) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  
  %0 = "TFHE.add_glwe"(%a0, %a1): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  return %0: !TFHE.glwe<sk?>
}

// -----

// CHECK:      func.func @fwd2(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>, %arg2: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:    %0 = "TFHE.add_glwe"(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:    %1 = "TFHE.add_glwe"(%0, %arg2) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:    return %1 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:  }
func.func @fwd2(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>, %arg2: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk?> {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a1 = "TypeInference.propagate_downward"(%arg1) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a2 = "TypeInference.propagate_downward"(%arg2) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  
  %0 = "TFHE.add_glwe"(%a0, %a1): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  %1 = "TFHE.add_glwe"(%0, %a2): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)

  return %1: !TFHE.glwe<sk?>
}

// -----

// CHECK:      func.func @for1(%arg0: index) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:   %1 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %0) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:     %2 = "TFHE.zero"() : () -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:     %inserted = tensor.insert %2 into %arg2[%arg1] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:     scf.yield %inserted : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:   }
// CHECK-NEXT:   %extracted = tensor.extract %1[%c0] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:   return %extracted : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT: }
func.func @for1(%idx: index) -> !TFHE.glwe<sk?> {
  %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk?>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iterarg = %0) -> (tensor<2x!TFHE.glwe<sk?>>) {
    %2 = "TFHE.zero"(): () -> (!TFHE.glwe<sk[1]<12,1024>>)
    %a = "TypeInference.propagate_downward"(%2) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
    %3 = tensor.insert %a into %iterarg[%i] : tensor<2x!TFHE.glwe<sk?>>
    scf.yield %3 : tensor<2x!TFHE.glwe<sk?>>
  }

  %4 = tensor.extract %1[%c0] : tensor<2x!TFHE.glwe<sk?>>
    
  return %4: !TFHE.glwe<sk?>
}

// -----

// CHECK:      func.func @for2(%arg0: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>, %arg1: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> tensor<2x!TFHE.glwe<sk[1]<12,1024>>> {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:      %extracted = tensor.extract %arg0[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      %extracted_0 = tensor.extract %arg1[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      %2 = "TFHE.add_glwe"(%extracted, %extracted_0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:      %inserted = tensor.insert %2 into %arg3[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      scf.yield %inserted : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:  }
func.func @for2(%arg0: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>, %arg1: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> tensor<2x!TFHE.glwe<sk?>> {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> (tensor<2x!TFHE.glwe<sk?>>)
  %a1 = "TypeInference.propagate_downward"(%arg1) : (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> (tensor<2x!TFHE.glwe<sk?>>)

  %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk?>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iterarg = %0) -> (tensor<2x!TFHE.glwe<sk?>>) {
    %2 = tensor.extract %a0[%i] : tensor<2x!TFHE.glwe<sk?>>
    %3 = tensor.extract %a1[%i] : tensor<2x!TFHE.glwe<sk?>>
    %4 = "TFHE.add_glwe"(%2, %3): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
    %5 = tensor.insert %4 into %iterarg[%i] : tensor<2x!TFHE.glwe<sk?>>
    scf.yield %5 : tensor<2x!TFHE.glwe<sk?>>
  }
    
  return %1: tensor<2x!TFHE.glwe<sk?>>
}

// -----

// CHECK:       func.func @for3(%arg0: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>, %arg1: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> tensor<2x!TFHE.glwe<sk[1]<12,1024>>> {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:      %extracted = tensor.extract %arg0[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      %extracted_0 = tensor.extract %arg1[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      %2 = "TFHE.add_glwe"(%extracted, %extracted_0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:      %inserted = tensor.insert %2 into %arg3[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      scf.yield %inserted : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:  }

func.func @for3(%arg0: tensor<2x!TFHE.glwe<sk?>>, %arg1: tensor<2x!TFHE.glwe<sk?>>) -> tensor<2x!TFHE.glwe<sk[1]<12,1024>>> {
  %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk?>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iterarg = %0) -> (tensor<2x!TFHE.glwe<sk?>>) {
    %2 = tensor.extract %arg0[%i] : tensor<2x!TFHE.glwe<sk?>>
    %3 = tensor.extract %arg1[%i] : tensor<2x!TFHE.glwe<sk?>>
    %4 = "TFHE.add_glwe"(%2, %3): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
    %5 = tensor.insert %4 into %iterarg[%i] : tensor<2x!TFHE.glwe<sk?>>
    scf.yield %5 : tensor<2x!TFHE.glwe<sk?>>
  }

  %r = "TypeInference.propagate_upward"(%1) : (tensor<2x!TFHE.glwe<sk?>>) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>)

  return %r: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
}

// -----

// CHECK:       func.func @for4(%arg0: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>, %arg1: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> tensor<2x!TFHE.glwe<sk[1]<12,1024>>> {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:      %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:        %extracted = tensor.extract %arg0[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:        %extracted_0 = tensor.extract %arg1[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:        %3 = "TFHE.add_glwe"(%extracted, %extracted_0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:        %inserted = tensor.insert %3 into %arg5[%arg4] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:        scf.yield %inserted : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %2 : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:  }

func.func @for4(%arg0: tensor<2x!TFHE.glwe<sk?>>, %arg1: tensor<2x!TFHE.glwe<sk?>>) -> tensor<2x!TFHE.glwe<sk[1]<12,1024>>> {
  %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk?>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iterarg0 = %0) -> (tensor<2x!TFHE.glwe<sk?>>) {
    %2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iterarg1 = %iterarg0) -> (tensor<2x!TFHE.glwe<sk?>>) {
      %3 = tensor.extract %arg0[%i] : tensor<2x!TFHE.glwe<sk?>>
      %4 = tensor.extract %arg1[%i] : tensor<2x!TFHE.glwe<sk?>>
      %5 = "TFHE.add_glwe"(%3, %4): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
      %6 = tensor.insert %5 into %iterarg1[%j] : tensor<2x!TFHE.glwe<sk?>>
      scf.yield %6 : tensor<2x!TFHE.glwe<sk?>>
    }

    scf.yield %2 : tensor<2x!TFHE.glwe<sk?>>
  }

  %r = "TypeInference.propagate_upward"(%1) : (tensor<2x!TFHE.glwe<sk?>>) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>)

  return %r: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
}

// -----

// CHECK:       func.func @for5(%arg0: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>, %arg1: tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> tensor<2x!TFHE.glwe<sk[1]<12,1024>>> {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    %1 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %0) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:      %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) {
// CHECK-NEXT:        %extracted = tensor.extract %arg0[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:        %extracted_0 = tensor.extract %arg1[%arg2] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:        %3 = "TFHE.add_glwe"(%extracted, %extracted_0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:        %inserted = tensor.insert %3 into %arg5[%arg4] : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:        scf.yield %inserted : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %2 : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %1 : tensor<2x!TFHE.glwe<sk[1]<12,1024>>>
// CHECK-NEXT:  }

func.func @for5(%arg0: tensor<2x!TFHE.glwe<sk?>>, %arg1: tensor<2x!TFHE.glwe<sk?>>) -> tensor<2x!TFHE.glwe<sk?>> {
  %0 = bufferization.alloc_tensor() : tensor<2x!TFHE.glwe<sk?>>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iterarg0 = %0) -> (tensor<2x!TFHE.glwe<sk?>>) {
    %2 = scf.for %j = %c0 to %c2 step %c1 iter_args(%iterarg1 = %iterarg0) -> (tensor<2x!TFHE.glwe<sk?>>) {
      %3 = tensor.extract %arg0[%i] : tensor<2x!TFHE.glwe<sk?>>
      %4 = tensor.extract %arg1[%i] : tensor<2x!TFHE.glwe<sk?>>
      %5 = "TFHE.add_glwe"(%3, %4): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
      %fivea = "TypeInference.propagate_upward"(%5) : (!TFHE.glwe<sk?>) -> (!TFHE.glwe<sk[1]<12,1024>>)
      %iterarg1a = "TypeInference.propagate_upward"(%iterarg1) : (tensor<2x!TFHE.glwe<sk?>>) -> (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>)
      %6 = tensor.insert %fivea into %iterarg1a[%j] : tensor<2x!TFHE.glwe<sk[1]<12, 1024>>>
      %sixa = "TypeInference.propagate_downward"(%6) : (tensor<2x!TFHE.glwe<sk[1]<12,1024>>>) -> (tensor<2x!TFHE.glwe<sk?>>)
      scf.yield %sixa : tensor<2x!TFHE.glwe<sk?>>
    }

    scf.yield %2 : tensor<2x!TFHE.glwe<sk?>>
  }

  return %1: tensor<2x!TFHE.glwe<sk?>>
}

// -----

#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>

func.func @tiled_2(%arg0: tensor<8x4x!TFHE.glwe<sk?>>, %arg1: tensor<4x2xi7>) -> tensor<8x2x!TFHE.glwe<sk?>> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %0 = "TFHE.zero_tensor"() : () -> tensor<8x2x!TFHE.glwe<sk?>>
  %1 = "TFHE.zero"() : () -> !TFHE.glwe<sk?>
  %2 = tensor.empty() : tensor<8x2x2x!TFHE.glwe<sk?>>
  %3 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %2) -> (tensor<8x2x2x!TFHE.glwe<sk?>>) {
    %6 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<8x2x2x!TFHE.glwe<sk?>>) {
      %7 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x2x2x!TFHE.glwe<sk?>>) {
        %inserted = tensor.insert %1 into %arg7[%arg2, %arg4, %arg6] : tensor<8x2x2x!TFHE.glwe<sk?>>
        scf.yield %inserted : tensor<8x2x2x!TFHE.glwe<sk?>>
      }
      scf.yield %7 : tensor<8x2x2x!TFHE.glwe<sk?>>
    }
    scf.yield %6 : tensor<8x2x2x!TFHE.glwe<sk?>>
  }
  %4 = scf.forall (%arg2) in (2) shared_outs(%arg3 = %3) -> (tensor<8x2x2x!TFHE.glwe<sk?>>) {
    %extracted_slice = tensor.extract_slice %arg3[0, 0, %arg2] [8, 2, 1] [1, 1, 1] : tensor<8x2x2x!TFHE.glwe<sk?>> to tensor<8x2x!TFHE.glwe<sk?>>
    %6 = affine.apply #map()[%arg2, %c2]
    %7 = affine.apply #map1()[%6, %c0]
    %8 = scf.for %arg4 = %7 to %c4 step %c4 iter_args(%arg5 = %extracted_slice) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
      %extracted_slice_0 = tensor.extract_slice %arg0[0, %arg4] [8, 2] [1, 1] : tensor<8x4x!TFHE.glwe<sk?>> to tensor<8x2x!TFHE.glwe<sk?>>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg4, 0] [2, 2] [1, 1] : tensor<4x2xi7> to tensor<2x2xi7>
      %9 = scf.for %arg6 = %c0 to %c8 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
        %10 = scf.for %arg8 = %c0 to %c2 step %c1 iter_args(%arg9 = %arg7) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
          %11 = scf.for %arg10 = %c0 to %c2 step %c1 iter_args(%arg11 = %arg9) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
            %extracted = tensor.extract %extracted_slice_0[%arg6, %arg10] : tensor<8x2x!TFHE.glwe<sk?>>
            %extracted_2 = tensor.extract %extracted_slice_1[%arg10, %arg8] : tensor<2x2xi7>
            %extracted_3 = tensor.extract %arg11[%arg6, %arg8] : tensor<8x2x!TFHE.glwe<sk?>>
            %12 = arith.extsi %extracted_2 : i7 to i64
            %13 = "TFHE.mul_glwe_int"(%extracted, %12) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>

	    // "Seed" type
	    %a13 = "TypeInference.propagate_upward"(%13) : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk[1]<12, 1024>>
            %aextracted_3 = "TypeInference.propagate_upward"(%extracted_3) : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk[1]<12, 1024>>
            %14 = "TFHE.add_glwe"(%aextracted_3, %a13) : (!TFHE.glwe<sk[1]<12, 1024>>, !TFHE.glwe<sk[1]<12, 1024>>) -> !TFHE.glwe<sk[1]<12, 1024>>
	    %a14 = "TypeInference.propagate_downward"(%14) : (!TFHE.glwe<sk[1]<12, 1024>>) -> !TFHE.glwe<sk?>

            %inserted = tensor.insert %a14 into %arg11[%arg6, %arg8] : tensor<8x2x!TFHE.glwe<sk?>>
            scf.yield %inserted : tensor<8x2x!TFHE.glwe<sk?>>
          }
          scf.yield %11 : tensor<8x2x!TFHE.glwe<sk?>>
        }
        scf.yield %10 : tensor<8x2x!TFHE.glwe<sk?>>
      }
      scf.yield %9 : tensor<8x2x!TFHE.glwe<sk?>>
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg3[0, 0, %arg2] [8, 2, 1] [1, 1, 1] : tensor<8x2x!TFHE.glwe<sk?>> into tensor<8x2x2x!TFHE.glwe<sk?>>
    }
  }
  %5 = scf.for %arg2 = %c0 to %c8 step %c1 iter_args(%arg3 = %0) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
    %6 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
      %7 = scf.for %arg6 = %c0 to %c2 step %c1 iter_args(%arg7 = %arg5) -> (tensor<8x2x!TFHE.glwe<sk?>>) {
        %extracted = tensor.extract %4[%arg2, %arg4, %arg6] : tensor<8x2x2x!TFHE.glwe<sk?>>
        %extracted_0 = tensor.extract %arg7[%arg2, %arg4] : tensor<8x2x!TFHE.glwe<sk?>>
        %8 = "TFHE.add_glwe"(%extracted, %extracted_0) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
        %inserted = tensor.insert %8 into %arg7[%arg2, %arg4] : tensor<8x2x!TFHE.glwe<sk?>>
        scf.yield %inserted : tensor<8x2x!TFHE.glwe<sk?>>
      }
      scf.yield %7 : tensor<8x2x!TFHE.glwe<sk?>>
    }
    scf.yield %6 : tensor<8x2x!TFHE.glwe<sk?>>
  }
  return %5 : tensor<8x2x!TFHE.glwe<sk?>>
}

// -----

// CHECK:      func.func @multi_block(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>, %arg2: !TFHE.glwe<sk[1]<12,1024>>, %arg3: i1, %arg4: i1) -> !TFHE.glwe<sk[1]<12,1024>> {
// CHECK-NEXT:   cf.cond_br %arg3, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   cf.cond_br %arg4, ^bb1, ^bb2
// CHECK-NEXT: ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT:   %0 = "TFHE.add_glwe"(%arg1, %arg0) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT:   return %0 : !TFHE.glwe<sk[1]<12,1024>>
// CHECK-NEXT: }
func.func @multi_block(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>, %arg2: !TFHE.glwe<sk[1]<12,1024>>, %cond: i1, %cond2: i1) -> !TFHE.glwe<sk?> {
  %a0 = "TypeInference.propagate_downward"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a1 = "TypeInference.propagate_downward"(%arg1) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  %a2 = "TypeInference.propagate_downward"(%arg2) : (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk?>)
  cf.cond_br %cond, ^bb0(%a0: !TFHE.glwe<sk?>), ^bb1(%a1: !TFHE.glwe<sk?>)
^bb0(%bbarg0 : !TFHE.glwe<sk?>):
  %0 = "TFHE.add_glwe"(%a1, %a1): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  cf.cond_br %cond2, ^bb0(%0: !TFHE.glwe<sk?>), ^bb1(%0: !TFHE.glwe<sk?>)
^bb1(%bbarg1 : !TFHE.glwe<sk?>):
  %1 = "TFHE.add_glwe"(%a1, %a0): (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> (!TFHE.glwe<sk?>)
  return %1: !TFHE.glwe<sk?>
}
