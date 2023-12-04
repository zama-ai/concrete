// RUN: concretecompiler --passes canonicalize --passes linalg-generalize-named-ops --passes MANP --passes ConcreteOptimizer --action=dump-fhe-no-linalg --split-input-file %s 2>&1 | FileCheck %s

func.func @conv2d_const_weight_const_bias(%input: tensor<1x1x4x4x!FHE.eint<6>>) -> tensor<1x1x2x2x!FHE.eint<6>> {
  %weight = arith.constant dense<[[[[1, 2], [2, 1]]]]> : tensor<1x1x2x2xi7>
  %bias = arith.constant dense<[5]> : tensor<1xi7>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %0 = "FHELinalg.conv2d"(%input, %weight, %bias){
    strides = dense<[2,2]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0,0,0]> : tensor<4xi64>
  } : (tensor<1x1x4x4x!FHE.eint<6>>, tensor<1x1x2x2xi7>, tensor<1xi7>) -> tensor<1x1x2x2x!FHE.eint<6>>
  return %0 : tensor<1x1x2x2x!FHE.eint<6>>
}

// -----

func.func @conv2d_const_weight(%input: tensor<1x1x4x4x!FHE.eint<6>>, %bias : tensor<1xi7>) -> tensor<1x1x2x2x!FHE.eint<6>> {
  %weight = arith.constant dense<[[[[1, 2], [2, 1]]]]> : tensor<1x1x2x2xi7>
  // CHECK: MANP = 4 : ui{{[0-9]+}}
  %0 = "FHELinalg.conv2d"(%input, %weight, %bias){
    strides = dense<[2,2]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0,0,0]> : tensor<4xi64>
  } : (tensor<1x1x4x4x!FHE.eint<6>>, tensor<1x1x2x2xi7>, tensor<1xi7>) -> tensor<1x1x2x2x!FHE.eint<6>>
  return %0 : tensor<1x1x2x2x!FHE.eint<6>>
}

// -----

func.func @conv2d_const_bias(%input: tensor<1x1x4x4x!FHE.eint<2>>, %weight: tensor<1x1x2x2xi3>) -> tensor<1x1x2x2x!FHE.eint<2>> {
  %bias = arith.constant dense<[5]> : tensor<1xi3>
  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %0 = "FHELinalg.conv2d"(%input, %weight, %bias){
    strides = dense<[2,2]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0,0,0]> : tensor<4xi64>
  } : (tensor<1x1x4x4x!FHE.eint<2>>, tensor<1x1x2x2xi3>, tensor<1xi3>) -> tensor<1x1x2x2x!FHE.eint<2>>
  return %0 : tensor<1x1x2x2x!FHE.eint<2>>
}

// -----

func.func @conv2d_weight_const_bias(%input: tensor<1x1x4x4x!FHE.eint<2>>, %weight: tensor<1x1x2x2xi3>, %bias : tensor<1xi3>) -> tensor<1x1x2x2x!FHE.eint<2>> {
  // CHECK: MANP = 6 : ui{{[0-9]+}}
  %0 = "FHELinalg.conv2d"(%input, %weight, %bias){
    strides = dense<[2,2]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0,0,0]> : tensor<4xi64>
  } : (tensor<1x1x4x4x!FHE.eint<2>>, tensor<1x1x2x2xi3>, tensor<1xi3>) -> tensor<1x1x2x2x!FHE.eint<2>>
  return %0 : tensor<1x1x2x2x!FHE.eint<2>>
}

// -----

func.func @conv2d_batched_multiple_channels(%input: tensor<100x3x4x4x!FHE.eint<2>>, %weight: tensor<5x3x2x2xi3>, %bias : tensor<5xi3>) -> tensor<100x5x2x2x!FHE.eint<2>> {
  // CHECK: MANP = 11 : ui{{[0-9]+}}
  %0 = "FHELinalg.conv2d"(%input, %weight, %bias){
    strides = dense<[2,2]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0,0,0,0]> : tensor<4xi64>
  } : (tensor<100x3x4x4x!FHE.eint<2>>, tensor<5x3x2x2xi3>, tensor<5xi3>) -> tensor<100x5x2x2x!FHE.eint<2>>
  return %0 : tensor<100x5x2x2x!FHE.eint<2>>
}

