#ifndef MLIR_DIALECT_RT_TRANSFORMS_PASSES
#define MLIR_DIALECT_RT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def HoistAwaitFuturePass : Pass<"hoist-await-future", "mlir::func::FuncOp"> {
  let summary = "Hoists `RT.await_future` operations whose results are yielded "
                "by `scf.forall` operations out of the loops";
  let description = [{
    Hoists `RT.await_future` operations whose results are yielded by
    scf.forall operations out of the loops in order to avoid
    over-synchronization of data-flow tasks.

    E.g., the following IR:

    ```
    scf.forall (%arg) in (16)
      shared_outs(%o1 = %sometensor, %o2 = %someothertensor)
      -> (tensor<...>, tensor<...>)
    {
      ...
      %rph = "RT.build_return_ptr_placeholder"() :
        () -> !RT.rtptr<!RT.future<tensor<...>>>
      "RT.create_async_task"(..., %rph, ...) { ... } : ...
      %future = "RT.deref_return_ptr_placeholder"(%rph) :
        (!RT.rtptr<!RT.future<...>>) -> !RT.future<tensor<...>>
      %res = "RT.await_future"(%future) : (!RT.future<tensor<...>>) -> tensor<...>
      ...
      scf.forall.in_parallel {
        ...
        tensor.parallel_insert_slice %res into %o1[..., %arg2, ...] [...] [...] :
          tensor<...> into tensor<...>
        ...
      }
    }
    ```

    is transformed into:

    ```
    %tensoroffutures = tensor.empty() : tensor<16x!RT.future<tensor<...>>>

    scf.forall (%arg) in (16)
      shared_outs(%otfut = %tensoroffutures, %o2 = %someothertensor)
      -> (tensor<...>, tensor<...>)
    {
      ...
      %rph = "RT.build_return_ptr_placeholder"() :
        () -> !RT.rtptr<!RT.future<tensor<...>>>
      "RT.create_async_task"(..., %rph, ...) { ... } : ...
      %future = "RT.deref_return_ptr_placeholder"(%rph) :
        (!RT.rtptr<!RT.future<...>>) -> !RT.future<tensor<...>>
      %wrappedfuture = tensor.from_elements %future :
        tensor<1x!RT.future<tensor<...>>>
      ...
      scf.forall.in_parallel {
        ...
        tensor.parallel_insert_slice %wrappedfuture into %otfut[%arg] [1] [1] :
          tensor<1xRT.future<tensor<...>>> into tensor<16x!RT.future<tensor<...>>>
        ...
      }
    }

    scf.forall (%arg) in (16) shared_outs(%o = %sometensor) -> (tensor<...>) {
      %future = tensor.extract %tensoroffutures[%arg] :
        tensor<4x!RT.future<tensor<...>>>
      %res = "RT.await_future"(%future) : (!RT.future<tensor<...>>) -> tensor<...>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %res into %o[..., %arg, ...] [...] [...] :
          tensor<...> into tensor<...>
      }
    }
    ```
  }];
  let constructor = "mlir::concretelang::createHoistAwaitFuturePass()";
  let dependentDialects = [
    "mlir::func::FuncDialect", "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect", "mlir::concretelang::RT::RTDialect"
  ];
}

#endif // MLIR_DIALECT_RT_TRANSFORMS_PASSES
