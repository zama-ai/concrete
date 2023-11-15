// RUN: not concretecompiler --split-input-file --action=roundtrip  %s 2>&1| FileCheck %s

// CHECK-LABEL: error: 'FHELinalg.lsb' op operand #0 must be , but got '!FHE.eint<2>
func.func @bad_tensor(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  %1 = "FHELinalg.lsb"(%arg0): (!FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// -----
