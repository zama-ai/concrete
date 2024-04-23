// RUN: concretecompiler --action=dump-fhe-df-parallelized %s --optimizer-strategy=dag-mono --parallelize --passes hoist-await-future --skip-program-info | FileCheck  %s

func.func @_dfr_DFT_work_function__main0(%arg0: !RT.rtptr<tensor<2x!FHE.eint<6>>>, %arg1: !RT.rtptr<tensor<2x!FHE.eint<6>>>, %arg2: !RT.rtptr<tensor<2xi7>>, %arg3: !RT.rtptr<tensor<2x!FHE.eint<6>>>) attributes {_dfr_work_function_attribute} {
  return
}

// CHECK:      %[[V3:.*]] = scf.forall (%[[Varg2:.*]]) in (8) shared_outs(%[[Varg3:.*]] = %[[V0:.*]]) -> (tensor<16x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[Vextracted:.*]] = tensor.extract %[[V2:.*]]{{\[}}%[[Varg2]]{{\]}} : tensor<8x!RT.future<tensor<2x!FHE.eint<6>>>>
// CHECK-NEXT:       %[[V4:.*]] = "RT.await_future"(%[[Vextracted]]) : (!RT.future<tensor<2x!FHE.eint<6>>>) -> tensor<2x!FHE.eint<6>>
// CHECK-NEXT:       %[[V5:.*]] = affine.apply #map(%[[Varg2]])
// CHECK-NEXT:       scf.forall.in_parallel {
// CHECK-NEXT:         tensor.parallel_insert_slice %[[V4]] into %[[Varg3]]{{\[}}%[[V5]]{{\] \[2\] \[1\]}} : tensor<2x!FHE.eint<6>> into tensor<16x!FHE.eint<6>>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V3]] : tensor<16x!FHE.eint<6>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @main(%arg0: tensor<16x!FHE.eint<6>>, %arg1: tensor<16xi7>) -> tensor<16x!FHE.eint<6>> {
  %f = constant @_dfr_DFT_work_function__main0 : (!RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2xi7>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>) -> ()
  "RT.register_task_work_function"(%f) : ((!RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2xi7>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>) -> ()) -> ()
  %0 = "FHE.zero_tensor"() : () -> tensor<16x!FHE.eint<6>>
  %1 = scf.forall (%arg2) in (8) shared_outs(%arg3 = %0) -> (tensor<16x!FHE.eint<6>>) {
    %2 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg2)
    %3 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg2)
    %4 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg2)
    %extracted_slice = tensor.extract_slice %arg0[%2] [2] [1] : tensor<16x!FHE.eint<6>> to tensor<2x!FHE.eint<6>>
    %extracted_slice_0 = tensor.extract_slice %arg1[%3] [2] [1] : tensor<16xi7> to tensor<2xi7>
    %extracted_slice_1 = tensor.extract_slice %arg3[%4] [2] [1] : tensor<16x!FHE.eint<6>> to tensor<2x!FHE.eint<6>>
    %c0_i64 = arith.constant 0 : i64
    %5 = "RT.make_ready_future"(%extracted_slice, %c0_i64) : (tensor<2x!FHE.eint<6>>, i64) -> !RT.future<tensor<2x!FHE.eint<6>>>
    %c0_i64_2 = arith.constant 0 : i64
    %6 = "RT.make_ready_future"(%extracted_slice_0, %c0_i64_2) : (tensor<2xi7>, i64) -> !RT.future<tensor<2xi7>>
    %c0_i64_3 = arith.constant 0 : i64
    %7 = "RT.make_ready_future"(%extracted_slice_1, %c0_i64_3) : (tensor<2x!FHE.eint<6>>, i64) -> !RT.future<tensor<2x!FHE.eint<6>>>
    %f_4 = func.constant @_dfr_DFT_work_function__main0 : (!RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2xi7>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>) -> ()
    %c3_i64 = arith.constant 3 : i64
    %c1_i64 = arith.constant 1 : i64
    %8 = "RT.build_return_ptr_placeholder"() : () -> !RT.rtptr<!RT.future<tensor<2x!FHE.eint<6>>>>
    "RT.create_async_task"(%f_4, %c3_i64, %c1_i64, %8, %5, %6, %7) {workfn = @_dfr_DFT_work_function__main0} : ((!RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>, !RT.rtptr<tensor<2xi7>>, !RT.rtptr<tensor<2x!FHE.eint<6>>>) -> (), i64, i64, !RT.rtptr<!RT.future<tensor<2x!FHE.eint<6>>>>, !RT.future<tensor<2x!FHE.eint<6>>>, !RT.future<tensor<2xi7>>, !RT.future<tensor<2x!FHE.eint<6>>>) -> ()
    %9 = "RT.deref_return_ptr_placeholder"(%8) : (!RT.rtptr<!RT.future<tensor<2x!FHE.eint<6>>>>) -> !RT.future<tensor<2x!FHE.eint<6>>>
    %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg2)
    %11 = "RT.await_future"(%9) : (!RT.future<tensor<2x!FHE.eint<6>>>) -> tensor<2x!FHE.eint<6>>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg3[%10] [2] [1] : tensor<2x!FHE.eint<6>> into tensor<16x!FHE.eint<6>>
    }
  }
  return %1 : tensor<16x!FHE.eint<6>>
}
