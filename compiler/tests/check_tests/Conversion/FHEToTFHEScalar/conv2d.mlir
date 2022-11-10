// RUN: concretecompiler %s --action=dump-tfhe 2>&1| FileCheck %s

//CHECK: func.func @conv2d(%arg0: tensor<100x3x28x28x!TFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<4x3x14x14xi3>, %arg2: tensor<4xi3>) -> tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:    %c4 = arith.constant 4 : index
// CHECK-NEXT:    %c100 = arith.constant 100 : index
// CHECK-NEXT:    %c15 = arith.constant 15 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c14 = arith.constant 14 : index
// CHECK-NEXT:    %0 = "TFHE.zero_tensor"() : () -> tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:    %1 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %0) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:      %3 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:        %4 = scf.for %arg7 = %c0 to %c15 step %c1 iter_args(%arg8 = %arg6) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:          %5 = scf.for %arg9 = %c0 to %c15 step %c1 iter_args(%arg10 = %arg8) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:            %6 = tensor.extract %arg2[%arg5] : tensor<4xi3>
// CHECK-NEXT:            %7 = tensor.extract %0[%arg3, %arg5, %arg7, %arg9] : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:            %8 = arith.extui %6 : i3 to i64
// CHECK-NEXT:            %c61_i64 = arith.constant 61 : i64
// CHECK-NEXT:            %9 = arith.shli %8, %c61_i64 : i64
// CHECK-NEXT:            %10 = "TFHE.add_glwe_int"(%7, %9) : (!TFHE.glwe<{_,_,_}{2}>, i64) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:            %11 = tensor.insert %10 into %arg10[%arg3, %arg5, %arg7, %arg9] : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:            scf.yield %11 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %5 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %4 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %3 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:    }
// CHECK-NEXT:    %2 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %1) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:      %3 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %arg4) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:        %4 = scf.for %arg7 = %c0 to %c15 step %c1 iter_args(%arg8 = %arg6) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:          %5 = scf.for %arg9 = %c0 to %c15 step %c1 iter_args(%arg10 = %arg8) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:            %6 = scf.for %arg11 = %c0 to %c3 step %c1 iter_args(%arg12 = %arg10) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:              %7 = scf.for %arg13 = %c0 to %c14 step %c1 iter_args(%arg14 = %arg12) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:                %8 = scf.for %arg15 = %c0 to %c14 step %c1 iter_args(%arg16 = %arg14) -> (tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:                  %9 = affine.apply #map(%arg7, %arg13)
// CHECK-NEXT:                  %10 = affine.apply #map(%arg9, %arg15)
// CHECK-NEXT:                  %11 = tensor.extract %arg0[%arg3, %arg11, %9, %10] : tensor<100x3x28x28x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:                  %12 = tensor.extract %arg1[%arg5, %arg11, %arg13, %arg15] : tensor<4x3x14x14xi3>
// CHECK-NEXT:                  %13 = tensor.extract %1[%arg3, %arg5, %arg7, %arg9] : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:                  %14 = arith.extsi %12 : i3 to i64
// CHECK-NEXT:                  %15 = "TFHE.mul_glwe_int"(%11, %14) : (!TFHE.glwe<{_,_,_}{2}>, i64) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:                  %16 = "TFHE.add_glwe"(%13, %15) : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:                  %17 = tensor.insert %16 into %arg16[%arg3, %arg5, %arg7, %arg9] : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:                  scf.yield %17 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:                }
// CHECK-NEXT:                scf.yield %8 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:              }
// CHECK-NEXT:              scf.yield %7 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %6 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %5 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %4 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %3 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %2 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
func.func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0, 0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}
