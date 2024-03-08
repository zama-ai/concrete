// RUN: concretecompiler %s --optimize-tfhe=false --optimizer-strategy=dag-mono --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @neg_eint(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
func.func @neg_eint(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT:  %0 = "TFHE.neg_glwe"(%arg0) : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  return %0 : !TFHE.glwe<sk?>

  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}
