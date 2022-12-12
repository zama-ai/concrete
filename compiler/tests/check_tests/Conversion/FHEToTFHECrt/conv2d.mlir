// RUN: concretecompiler --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s


//CHECK-LABEL:  func.func @conv2d(%arg0: tensor<100x3x28x28x5x!TFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<4x3x14x14xi3>, %arg2: tensor<4xi3>) -> tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>> {
//CHECK-NEXT:    %c4 = arith.constant 4 : index
//CHECK-NEXT:    %c100 = arith.constant 100 : index
//CHECK-NEXT:    %c15 = arith.constant 15 : index
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    %c3 = arith.constant 3 : index
//CHECK-NEXT:    %c14 = arith.constant 14 : index
//CHECK-NEXT:    %0 = "TFHE.zero_tensor"() : () -> tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:    %1 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %0) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:      %3 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:        %4 = scf.for %arg7 = %c0 to %c15 step %c1 iter_args(%arg8 = %arg6) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:          %5 = scf.for %arg9 = %c0 to %c15 step %c1 iter_args(%arg10 = %arg8) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:            %6 = tensor.extract %arg2[%arg5] : tensor<4xi3>
//CHECK-NEXT:            %c0_0 = arith.constant 0 : index
//CHECK-NEXT:            %7 = tensor.extract_slice %0[%arg3, %arg5, %arg7, %arg9, %c0_0] [1, 1, 1, 1, 5] [1, 1, 1, 1, 1] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>> to tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            %8 = arith.extui %6 : i3 to i64
//CHECK-NEXT:            %9 = "TFHE.encode_plaintext_with_crt"(%8) {mods = [2, 3, 5, 7, 11], modsProd = 2310 : i64} : (i64) -> tensor<5xi64>
//CHECK-NEXT:            %10 = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            %c0_1 = arith.constant 0 : index
//CHECK-NEXT:            %c1_2 = arith.constant 1 : index
//CHECK-NEXT:            %c5 = arith.constant 5 : index
//CHECK-NEXT:            %11 = scf.for %arg11 = %c0_1 to %c5 step %c1_2 iter_args(%arg12 = %10) -> (tensor<5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:              %13 = tensor.extract %7[%arg11] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:              %14 = tensor.extract %9[%arg11] : tensor<5xi64>
//CHECK-NEXT:              %15 = "TFHE.add_glwe_int"(%13, %14) : (!TFHE.glwe<{_,_,_}{2}>, i64) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:              %16 = tensor.insert %15 into %arg12[%arg11] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:              scf.yield %16 : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            }
//CHECK-NEXT:            %c0_3 = arith.constant 0 : index
//CHECK-NEXT:            %12 = tensor.insert_slice %11 into %arg10[%arg3, %arg5, %arg7, %arg9, %c0_3] [1, 1, 1, 1, 5] [1, 1, 1, 1, 1] : tensor<5x!TFHE.glwe<{_,_,_}{2}>> into tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            scf.yield %12 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:          }
//CHECK-NEXT:          scf.yield %5 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:        }
//CHECK-NEXT:        scf.yield %4 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:      }
//CHECK-NEXT:      scf.yield %3 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:    }
//CHECK-NEXT:    %2 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %1) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:      %3 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:        %4 = scf.for %arg7 = %c0 to %c15 step %c1 iter_args(%arg8 = %arg6) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:          %5 = scf.for %arg9 = %c0 to %c15 step %c1 iter_args(%arg10 = %arg8) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:            %6 = scf.for %arg11 = %c0 to %c3 step %c1 iter_args(%arg12 = %arg10) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:              %7 = scf.for %arg13 = %c0 to %c14 step %c1 iter_args(%arg14 = %arg12) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                %8 = scf.for %arg15 = %c0 to %c14 step %c1 iter_args(%arg16 = %arg14) -> (tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                  %9 = affine.apply #map(%arg7, %arg13)
//CHECK-NEXT:                  %10 = affine.apply #map(%arg9, %arg15)
//CHECK-NEXT:                  %c0_0 = arith.constant 0 : index
//CHECK-NEXT:                  %11 = tensor.extract_slice %arg0[%arg3, %arg11, %9, %10, %c0_0] [1, 1, 1, 1, 5] [1, 1, 1, 1, 1] : tensor<100x3x28x28x5x!TFHE.glwe<{_,_,_}{2}>> to tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %12 = tensor.extract %arg1[%arg5, %arg11, %arg13, %arg15] : tensor<4x3x14x14xi3>
//CHECK-NEXT:                  %c0_1 = arith.constant 0 : index
//CHECK-NEXT:                  %13 = tensor.extract_slice %1[%arg3, %arg5, %arg7, %arg9, %c0_1] [1, 1, 1, 1, 5] [1, 1, 1, 1, 1] : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>> to tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %14 = arith.extsi %12 : i3 to i64
//CHECK-NEXT:                  %15 = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %c0_2 = arith.constant 0 : index
//CHECK-NEXT:                  %c1_3 = arith.constant 1 : index
//CHECK-NEXT:                  %c5 = arith.constant 5 : index
//CHECK-NEXT:                  %16 = scf.for %arg17 = %c0_2 to %c5 step %c1_3 iter_args(%arg18 = %15) -> (tensor<5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                    %20 = tensor.extract %11[%arg17] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    %21 = "TFHE.mul_glwe_int"(%20, %14) : (!TFHE.glwe<{_,_,_}{2}>, i64) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:                    %22 = tensor.insert %21 into %arg18[%arg17] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    scf.yield %22 : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  }
//CHECK-NEXT:                  %17 = bufferization.alloc_tensor() : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  %c0_4 = arith.constant 0 : index
//CHECK-NEXT:                  %c1_5 = arith.constant 1 : index
//CHECK-NEXT:                  %c5_6 = arith.constant 5 : index
//CHECK-NEXT:                  %18 = scf.for %arg17 = %c0_4 to %c5_6 step %c1_5 iter_args(%arg18 = %17) -> (tensor<5x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:                    %20 = tensor.extract %13[%arg17] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    %21 = tensor.extract %16[%arg17] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    %22 = "TFHE.add_glwe"(%20, %21) : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:                    %23 = tensor.insert %22 into %arg18[%arg17] : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                    scf.yield %23 : tensor<5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  }
//CHECK-NEXT:                  %c0_7 = arith.constant 0 : index
//CHECK-NEXT:                  %19 = tensor.insert_slice %18 into %arg16[%arg3, %arg5, %arg7, %arg9, %c0_7] [1, 1, 1, 1, 5] [1, 1, 1, 1, 1] : tensor<5x!TFHE.glwe<{_,_,_}{2}>> into tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                  scf.yield %19 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:                }
//CHECK-NEXT:                scf.yield %8 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:              }
//CHECK-NEXT:              scf.yield %7 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:            }
//CHECK-NEXT:            scf.yield %6 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:          }
//CHECK-NEXT:          scf.yield %5 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:        }
//CHECK-NEXT:        scf.yield %4 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:      }
//CHECK-NEXT:      scf.yield %3 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:    }
//CHECK-NEXT:    return %2 : tensor<100x4x15x15x5x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:  }
func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0, 0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}
