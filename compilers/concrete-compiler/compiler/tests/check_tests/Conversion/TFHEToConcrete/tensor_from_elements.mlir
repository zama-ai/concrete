// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete --split-input-file --skip-program-info %s 2>&1| FileCheck %s

// CHECK: func.func @main(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<2049xi64>, %[[A2:.*]]: tensor<2049xi64>, %[[A3:.*]]: tensor<2049xi64>, %[[A4:.*]]: tensor<2049xi64>, %[[A5:.*]]: tensor<2049xi64>) -> tensor<6x2049xi64> {
// CHECK:   %[[V0:.*]] = bufferization.alloc_tensor() : tensor<6x2049xi64>
// CHECK:   %[[V1:.*]] = tensor.insert_slice %[[A0]] into %[[V0]][0, 0] [1, 2049] [1, 1] : tensor<2049xi64> into tensor<6x2049xi64>
// CHECK:   %[[V2:.*]] = tensor.insert_slice %[[A1]] into %[[V1]][1, 0] [1, 2049] [1, 1] : tensor<2049xi64> into tensor<6x2049xi64>
// CHECK:   %[[V3:.*]] = tensor.insert_slice %[[A2]] into %[[V2]][2, 0] [1, 2049] [1, 1] : tensor<2049xi64> into tensor<6x2049xi64>
// CHECK:   %[[V4:.*]] = tensor.insert_slice %[[A3]] into %[[V3]][3, 0] [1, 2049] [1, 1] : tensor<2049xi64> into tensor<6x2049xi64>
// CHECK:   %[[V5:.*]] = tensor.insert_slice %[[A4]] into %[[V4]][4, 0] [1, 2049] [1, 1] : tensor<2049xi64> into tensor<6x2049xi64>
// CHECK:   %[[V6:.*]] = tensor.insert_slice %[[A5]] into %[[V5]][5, 0] [1, 2049] [1, 1] : tensor<2049xi64> into tensor<6x2049xi64>
// CHECK:   return %[[V6]] : tensor<6x2049xi64>
// CHECK: }
func.func @main(%arg0 : !TFHE.glwe<sk[1]<1,2048>>, %arg1 : !TFHE.glwe<sk[1]<1,2048>>, %arg2 : !TFHE.glwe<sk[1]<1,2048>>, %arg3 : !TFHE.glwe<sk[1]<1,2048>>, %arg4 : !TFHE.glwe<sk[1]<1,2048>>, %arg5 : !TFHE.glwe<sk[1]<1,2048>>) -> tensor<6x!TFHE.glwe<sk[1]<1,2048>>> {
  %0 = tensor.from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<6x!TFHE.glwe<sk[1]<1,2048>>>
  return %0 : tensor<6x!TFHE.glwe<sk[1]<1,2048>>>
}

// -----

// CHECK: func.func @main(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<2049xi64>, %[[A2:.*]]: tensor<2049xi64>, %[[A3:.*]]: tensor<2049xi64>, %[[A4:.*]]: tensor<2049xi64>, %[[A5:.*]]: tensor<2049xi64>) -> tensor<2x3x2049xi64> {
// CHECK:   %[[V0:.*]] = bufferization.alloc_tensor() : tensor<2x3x2049xi64>
// CHECK:   %[[V1:.*]] = tensor.insert_slice %[[A0]] into %[[V0]][0, 0, 0] [1, 1, 2049] [1, 1, 1] : tensor<2049xi64> into tensor<2x3x2049xi64>
// CHECK:   %[[V2:.*]] = tensor.insert_slice %[[A1]] into %[[V1]][0, 1, 0] [1, 1, 2049] [1, 1, 1] : tensor<2049xi64> into tensor<2x3x2049xi64>
// CHECK:   %[[V3:.*]] = tensor.insert_slice %[[A2]] into %[[V2]][0, 2, 0] [1, 1, 2049] [1, 1, 1] : tensor<2049xi64> into tensor<2x3x2049xi64>
// CHECK:   %[[V4:.*]] = tensor.insert_slice %[[A3]] into %[[V3]][1, 0, 0] [1, 1, 2049] [1, 1, 1] : tensor<2049xi64> into tensor<2x3x2049xi64>
// CHECK:   %[[V5:.*]] = tensor.insert_slice %[[A4]] into %[[V4]][1, 1, 0] [1, 1, 2049] [1, 1, 1] : tensor<2049xi64> into tensor<2x3x2049xi64>
// CHECK:   %[[V6:.*]] = tensor.insert_slice %[[A5]] into %[[V5]][1, 2, 0] [1, 1, 2049] [1, 1, 1] : tensor<2049xi64> into tensor<2x3x2049xi64>
// CHECK:   return %[[V6]] : tensor<2x3x2049xi64>
// CHECK: }
func.func @main(%arg0 : !TFHE.glwe<sk[1]<1,2048>>, %arg1 : !TFHE.glwe<sk[1]<1,2048>>, %arg2 : !TFHE.glwe<sk[1]<1,2048>>, %arg3 : !TFHE.glwe<sk[1]<1,2048>>, %arg4 : !TFHE.glwe<sk[1]<1,2048>>, %arg5 : !TFHE.glwe<sk[1]<1,2048>>) -> tensor<2x3x!TFHE.glwe<sk[1]<1,2048>>> {
  %0 = tensor.from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<2x3x!TFHE.glwe<sk[1]<1,2048>>>
  return %0 : tensor<2x3x!TFHE.glwe<sk[1]<1,2048>>>
}
