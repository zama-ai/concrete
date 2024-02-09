// RUN: concretecompiler --split-input-file --passes tfhe-to-concrete --action=dump-concrete --skip-program-info %s 2>&1| FileCheck %s

//CHECK: func.func @tensor_collapse_shape(%[[A0:.*]]: tensor<2x3x4x5x6x1025xi64>) -> tensor<720x1025xi64> {
//CHECK:   %[[V0:.*]] = tensor.collapse_shape %[[A0]] [[_:\[\[0, 1, 2, 3, 4\], \[5\]\]]] : tensor<2x3x4x5x6x1025xi64> into tensor<720x1025xi64>
//CHECK:   return %[[V0]] : tensor<720x1025xi64>
//CHECK: }
func.func @tensor_collapse_shape(%arg0: tensor<2x3x4x5x6x!TFHE.glwe<sk[1]<1,1024>>>) -> tensor<720x!TFHE.glwe<sk[1]<1,1024>>> {
    %0 = tensor.collapse_shape %arg0 [[0, 1, 2, 3, 4]]  {MANP = 1 : ui1}: tensor<2x3x4x5x6x!TFHE.glwe<sk[1]<1,1024>>> into tensor<720x!TFHE.glwe<sk[1]<1,1024>>>
    return %0 : tensor<720x!TFHE.glwe<sk[1]<1,1024>>>
}

// -----
//CHECK: func.func @tensor_collatenspse_shape(%[[A0:.*]]: tensor<2x3x5x1025xi64>) -> tensor<5x6x1025xi64> {
//CHECK:   %[[V0:.*]] = tensor.collapse_shape %[[A0]] [[_:\[\[0, 1, 2\], \[3\]\]]] : tensor<2x3x5x1025xi64> into tensor<30x1025xi64>
//CHECK:   %[[V1:.*]] = tensor.expand_shape %[[V0]] [[_:\[\[0, 1\], \[2\]\]]] : tensor<30x1025xi64> into tensor<5x6x1025xi64>
//CHECK:   return %[[V1]] : tensor<5x6x1025xi64>
//CHECK: }
func.func @tensor_collatenspse_shape(%arg0: tensor<2x3x5x!TFHE.glwe<sk[1]<1,1024>>>) -> tensor<5x6x!TFHE.glwe<sk[1]<1,1024>>> {
    %0 = tensor.collapse_shape %arg0 [[0, 1, 2]]  {MANP = 1 : ui1}: tensor<2x3x5x!TFHE.glwe<sk[1]<1,1024>>> into tensor<30x!TFHE.glwe<sk[1]<1,1024>>>
    %1 = tensor.expand_shape %0 [[0, 1]]  {MANP = 1 : ui1}: tensor<30x!TFHE.glwe<sk[1]<1,1024>>> into tensor<5x6x!TFHE.glwe<sk[1]<1,1024>>>
    return %1 : tensor<5x6x!TFHE.glwe<sk[1]<1,1024>>>
}

// -----
//CHECK: func.func @tensor_collatenspse_shape(%[[A0:.*]]: tensor<2x3x2x3x4x1025xi64>) -> tensor<6x2x12x1025xi64> {
//CHECK:   %[[V0:.*]] = tensor.collapse_shape %[[A0]] [[_:\[\[0, 1\], \[2\], \[3, 4\], \[5\]\]]] : tensor<2x3x2x3x4x1025xi64> into tensor<6x2x12x1025xi64>
//CHECK:   return %[[V0]] : tensor<6x2x12x1025xi64>
//CHECK: }
func.func @tensor_collatenspse_shape(%arg0: tensor<2x3x2x3x4x!TFHE.glwe<sk[1]<1,1024>>>) -> tensor<6x2x12x!TFHE.glwe<sk[1]<1,1024>>> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2], [3, 4]]  {MANP = 1 : ui1}: tensor<2x3x2x3x4x!TFHE.glwe<sk[1]<1,1024>>> into tensor<6x2x12x!TFHE.glwe<sk[1]<1,1024>>>
    return %0 : tensor<6x2x12x!TFHE.glwe<sk[1]<1,1024>>>
}

// -----
//CHECK: func.func @tensor_expand_shape_crt(%[[A0:.*]]: tensor<30x1025xi64>) -> tensor<5x6x1025xi64> {
//CHECK:   %[[V0:.*]] = tensor.expand_shape %[[A0]] [[_:\[\[0, 1\], \[2\]\]]] : tensor<30x1025xi64> into tensor<5x6x1025xi64>
//CHECK:   return %[[V0]] : tensor<5x6x1025xi64>
//CHECK: }
func.func @tensor_expand_shape_crt(%arg0: tensor<30x!TFHE.glwe<sk[1]<1,1024>>>) -> tensor<5x6x!TFHE.glwe<sk[1]<1,1024>>> {
    %0 = tensor.expand_shape %arg0 [[0, 1]]  {MANP = 1 : ui1}: tensor<30x!TFHE.glwe<sk[1]<1,1024>>> into tensor<5x6x!TFHE.glwe<sk[1]<1,1024>>>
    return %0 : tensor<5x6x!TFHE.glwe<sk[1]<1,1024>>>
}
