// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHE.lsb' op operand #0 must be , but got 'tensor<1x!FHE.eint<2>>
func.func @bad_scalar(%arg0: tensor<1x!FHE.eint<2>>) -> !FHE.eint<2> {
  %1 = "FHE.lsb"(%arg0): (tensor<1x!FHE.eint<2>>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// -----
