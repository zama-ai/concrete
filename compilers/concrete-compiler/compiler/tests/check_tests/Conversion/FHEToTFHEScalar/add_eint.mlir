// RUN: concretecompiler %s --optimize-tfhe=false --optimizer-strategy=dag-mono --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_eint(%arg0: !TFHE.glwe<sk?>, %arg1: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
func.func @add_eint(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.add_glwe"(%arg0, %arg1) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<sk?>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
