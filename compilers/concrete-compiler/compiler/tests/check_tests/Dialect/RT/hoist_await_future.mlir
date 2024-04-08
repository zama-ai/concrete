// RUN: concretecompiler --action=dump-fhe-df-parallelized %s --optimizer-strategy=dag-mono --parallelize | FileCheck  %s
// RUN: concretecompiler --action=dump-llvm-ir %s --optimizer-strategy=dag-mono --parallelize
// RUN: concretecompiler --action=dump-llvm-ir %s --optimizer-strategy=dag-multi --parallelize

// CHECK:       scf.forall.in_parallel {
// CHECK-NEXT:    tensor.parallel_insert_slice %from_elements into %arg3[%arg2] [1] [1] : tensor<1x!RT.future<tensor<8x9x!FHE.eint<6>>>> into tensor<4x!RT.future<tensor<8x9x!FHE.eint<6>>>>
// CHECK-NEXT:  }
//
// CHECK:      %[[res:.*]] = scf.forall (%[[arg:.*]]) in (4) shared_outs(%[[so:.*]] = %[[init:.*]]) -> (tensor<8x9x4x!FHE.eint<6>>) {
// CHECK-NEXT:    %[[extracted:.*]] = tensor.extract %4[%[[arg]]] : tensor<4x!RT.future<tensor<8x9x!FHE.eint<6>>>>
// CHECK-NEXT:    %[[awaitres:.*]] = "RT.await_future"(%[[extracted]]) : (!RT.future<tensor<8x9x!FHE.eint<6>>>) -> tensor<8x9x!FHE.eint<6>>
// CHECK-NEXT:    scf.forall.in_parallel {
// CHECK-NEXT:      tensor.parallel_insert_slice %[[awaitres]] into %[[so]][0, 0, %[[arg]]] [8, 9, 1] [1, 1, 1] : tensor<8x9x!FHE.eint<6>> into tensor<8x9x4x!FHE.eint<6>>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

func.func @main(%a: tensor<8x7x!FHE.eint<6>>, %b: tensor<7x9xi7>) -> tensor<8x9x!FHE.eint<6>>{
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [0, 0, 2] } : (tensor<8x7x!FHE.eint<6>>, tensor<7x9xi7>) -> tensor<8x9x!FHE.eint<6>>
  return %0 : tensor<8x9x!FHE.eint<6>>
}
