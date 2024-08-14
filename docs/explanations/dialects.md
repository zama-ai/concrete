# Overview of FHE-related dialects in Concrete Compiler
## Introduction

Compilation of a Python program starts with Concrete's Python frontend, which first traces and transforms it and then converts it into an intermediate representation (IR) that is further processed by Concrete Compiler. This IR is based on the [MLIR subproject](https://mlir.llvm.org/) of the [LLVM compiler infrastructure](https://www.llvm.org). This document provides an overview of Concrete's FHE-specific representations based on the MLIR framework.

In contrast to traditional infrastructure for compilers, the set of operations and data types that constitute the IR, as well as the level of abstraction that the IR represents, are not fixed in MLIR and can easily be extended. All operations and data types are grouped into [dialects](https://mlir.llvm.org/docs/LangRef/#dialects), with each dialect representing a specific domain or a specific level of abstraction. Mixing operations and types from different dialects within the same IR is allowed and even encouraged, with all dialects--builtin or developed as an extension--being first-class citizens.

Concrete compiler takes advantage of these concepts by defining a set of dialects, capable of representing an FHE program from an abstract specification that is independent of the actual cryptosystem down to a program that can easily be mapped to function calls of a cryptographic library. The dialects for the representation of an FHE program are:

* The FHELinalg Dialect ([documentation](FHELinalgDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/FHELinalg/IR/FHELinalgOps.td))
* The FHE Dialect ([documentation](FHEDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/FHE/IR/FHEOps.td))
* The TFHE Dialect ([documentation](TFHEDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/TFHE/IR/TFHEOps.td))
* The Concrete Dialect ([documentation](ConcreteDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/Concrete/IR/ConcreteOps.td))
* and for debugging purposes, the Tracing Dialect ([documentation](TracingDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/Tracing/IR/TracingOps.td)).

In addition, the project further defines two dialects that help expose dynamic task-parallelism and static data-flow graphs in order to benefit from multi-core, multi-accelerator and distributed systems. These are:

* The RT Dialect ([documentation](RTDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/RT/IR/RTOps.td)) and
* The SDFG Dialect ([documentation](SDFGDialect.md), [source](../../compilers/concrete-compiler/compiler/include/concretelang/Dialect/SDFG/IR/SDFGOps.td)).

The figure below illustrates the relationship between the dialects and their embedding into the compilation pipeline.

![](../\_static/compilation-pipeline/compilation-flow.png)

The following sections focus on the FHE-related dialects, i.e., on the FHELinalg Dialect, the FHE Dialect, the TFHE Dialect and the Concrete Dialect.

## The FHE and FHELinalg Dialects: An abstract specification of an FHE program

The top part of the figure shows the components which are involved in the generation of the initial IR, ending with the step labelled *MLIR translation*. When the initial IR is passed on to Concrete Compiler through its Python bindings, all FHE-related operations are specified using either the FHE or FHELinalg Dialect. Both of these dialects provide operations and data types for the abstract specification of an FHE program, completely independently of a cryptosystem. At this point, the IR simply indicates whether an operand is encrypted (via the type `FHE.eint<n>`, where `n` stands for the precision in bits) and what operations are applied to encrypted values. Plaintext values simply use MLIR's builtin integer type `in` (e.g., `i3` or `i64`).

The FHE Dialect provides scalar operations on encrypted integers, such as additions (`FHE.add_eint`) or multiplications (`FHE.mul_eint`), while the FHELinalg Dialect offers operations on tensors of encrypted integers, e.g., matrix products (`FHELinalg.matmul_eint_eint`) or convolutions (`FHELinalg.conv2d`).

In a first lowering step of the pipeline, all FHELinalg operations are lowered to operations from [MLIR's builtin Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/) using scalar operations from the FHE Dialect. Consider the following example, which consists of a function that performs a multiplication of a matrix of encrypted integers and a matrix of cleartext values:

```mlir
func.func @main(%arg0: tensor<4x3x!FHE.eint<2>>, %arg1: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>>
  return %0 : tensor<4x2x!FHE.eint<2>>
}
```

Upon conversion, the `FHELinalg.matmul` operation is converted to a `linalg.generic` operation whose body contains a scalar multiplication (`FHE.mul_eint_int`) and a scalar addition (`FHE.add_eint_int`):


```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @main(%arg0: tensor<4x3x!FHE.eint<2>>, %arg1: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x2x!FHE.eint<2>>
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x3x!FHE.eint<2>>, tensor<3x2xi3>) outs(%0 : tensor<4x2x!FHE.eint<2>>) {
  ^bb0(%in: !FHE.eint<2>, %in_0: i3, %out: !FHE.eint<2>):
    %2 = "FHE.mul_eint_int"(%in, %in_0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
    %3 = "FHE.add_eint"(%out, %2) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    linalg.yield %3 : !FHE.eint<2>
  } -> tensor<4x2x!FHE.eint<2>>
  return %1 : tensor<4x2x!FHE.eint<2>>
}
```

This is then further lowered to a nest of loops from [MLIR's SCF Dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/), implementing the parallel and reduction dimensions from the `linalg.generic` operation above:

```mlir
func.func @main(%arg0: tensor<4x3x!FHE.eint<2>>, %arg1: tensor<3x2xi3>) -> tensor<4x2x!FHE.eint<2>> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = "FHE.zero_tensor"() : () -> tensor<4x2x!FHE.eint<2>>
  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x2x!FHE.eint<2>>) {
    %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x2x!FHE.eint<2>>) {
      %3 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x2x!FHE.eint<2>>) {
        %extracted = tensor.extract %arg0[%arg2, %arg6] : tensor<4x3x!FHE.eint<2>>
        %extracted_0 = tensor.extract %arg1[%arg6, %arg4] : tensor<3x2xi3>
        %extracted_1 = tensor.extract %arg7[%arg2, %arg4] : tensor<4x2x!FHE.eint<2>>
        %4 = "FHE.mul_eint_int"(%extracted, %extracted_0) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
        %5 = "FHE.add_eint"(%extracted_1, %4) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
        %inserted = tensor.insert %5 into %arg7[%arg2, %arg4] : tensor<4x2x!FHE.eint<2>>
        scf.yield %inserted : tensor<4x2x!FHE.eint<2>>
      }
      scf.yield %3 : tensor<4x2x!FHE.eint<2>>
    }
    scf.yield %2 : tensor<4x2x!FHE.eint<2>>
  }
  return %1 : tensor<4x2x!FHE.eint<2>>
}
```

## The TFHE Dialect: Binding to the TFHE cryptosystem and parametrization

In order to obtain an executable program at the end of the compilation pipeline, the abstract specification of the FHE program must at some point be bound to a specific cryptosystem. This is the role of the TFHE Dialect, whose purpose is:
* to indicate operations to be carried out using an implementation of the TFHE cryptosystem
* to parametrize the cryptosystem with key sizes, and
* to provide a mapping between keys and encrypted values

When lowering the IR based on the FHE Dialect to the TFHE Dialect, the compiler first generates a generic form, in which FHE operations are lowered to TFHE operations and where values are converted to unparametrized `TFHE.glwe` values. The unparametrized form `TFHE.glwe<sk?>` simply indicates that a `TFHE.glwe` value is to be used, but without any indication of the cryptographic parameters and the actual key.

The IR below shows the example program after lowering to unparametrized TFHE:

```mlir
func.func @main(%arg0: tensor<4x3x!TFHE.glwe<sk?>>, %arg1: tensor<3x2xi3>) -> tensor<4x2x!TFHE.glwe<sk?>> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = "TFHE.zero_tensor"() : () -> tensor<4x2x!TFHE.glwe<sk?>>
  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x2x!TFHE.glwe<sk?>>) {
    %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x2x!TFHE.glwe<sk?>>) {
      %3 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x2x!TFHE.glwe<sk?>>) {
        %extracted = tensor.extract %arg0[%arg2, %arg6] : tensor<4x3x!TFHE.glwe<sk?>>
        %extracted_0 = tensor.extract %arg1[%arg6, %arg4] : tensor<3x2xi3>
        %extracted_1 = tensor.extract %arg7[%arg2, %arg4] : tensor<4x2x!TFHE.glwe<sk?>>
        %4 = arith.extsi %extracted_0 : i3 to i64
        %5 = "TFHE.mul_glwe_int"(%extracted, %4) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
        %6 = "TFHE.add_glwe"(%extracted_1, %5) : (!TFHE.glwe<sk?>, !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
        %inserted = tensor.insert %6 into %arg7[%arg2, %arg4] : tensor<4x2x!TFHE.glwe<sk?>>
        scf.yield %inserted : tensor<4x2x!TFHE.glwe<sk?>>
      }
      scf.yield %3 : tensor<4x2x!TFHE.glwe<sk?>>
    }
    scf.yield %2 : tensor<4x2x!TFHE.glwe<sk?>>
  }
  return %1 : tensor<4x2x!TFHE.glwe<sk?>>
}
```

All operations from the FHE dialect have been replaced with corresponding operations from the TFHE Dialect.

During subsequent parametrization, the compiler can either use a set of default parameters or can obtain a set of parameters from Concrete's optimizer. Either way, an additional pass injects the parameters into the IR, replacing all `TFHE.glwe<sk?>` instances with `TFHE.glwe<i,d,n>`, where `i` is a sequential identifier for a key, `d` the number of GLWE dimensions and `n` the size of the GLWE polynomial.

The result of such a parametrization for the example is given below:

```mlir
func.func @main(%arg0: tensor<4x3x!TFHE.glwe<sk<0,1,512>>>, %arg1: tensor<3x2xi3>) -> tensor<4x2x!TFHE.glwe<sk<0,1,512>>> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %0 = "TFHE.zero_tensor"() : () -> tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<4x2x!TFHE.glwe<sk<0,1,512>>>) {
    %2 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x2x!TFHE.glwe<sk<0,1,512>>>) {
      %3 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x2x!TFHE.glwe<sk<0,1,512>>>) {
        %extracted = tensor.extract %arg0[%arg2, %arg6] : tensor<4x3x!TFHE.glwe<sk<0,1,512>>>
        %extracted_0 = tensor.extract %arg1[%arg6, %arg4] : tensor<3x2xi3>
        %extracted_1 = tensor.extract %arg7[%arg2, %arg4] : tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
        %4 = arith.extsi %extracted_0 : i3 to i64
        %5 = "TFHE.mul_glwe_int"(%extracted, %4) : (!TFHE.glwe<sk<0,1,512>>, i64) -> !TFHE.glwe<sk<0,1,512>>
        %6 = "TFHE.add_glwe"(%extracted_1, %5) : (!TFHE.glwe<sk<0,1,512>>, !TFHE.glwe<sk<0,1,512>>) -> !TFHE.glwe<sk<0,1,512>>
        %inserted = tensor.insert %6 into %arg7[%arg2, %arg4] : tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
        scf.yield %inserted : tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
      }
      scf.yield %3 : tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
    }
    scf.yield %2 : tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
  }
  return %1 : tensor<4x2x!TFHE.glwe<sk<0,1,512>>>
}
```

In this parametrization, a single key with the ID `0` is used, with a single dimension and a polynomial of size 512.

## The Concrete Dialect: Preparing bindings with a crypto library

In the next step of the pipeline, operations and types are lowered to the Concrete Dialect. This dialect provides operations, which are implemented by one of Concrete's backend libraries, but still abstracts from any technical details required for interaction with an actual library. The goal is to maintain a high-level representation with value-based semantics and actual operations instead of buffer semantics and library calls, while ensuring that all operations  an effectively be lowered to a library call later in the pipeline. However, the abstract types from TFHE are already lowered to tensors of integers with a suitable shape that will hold the binary data of the encrypted values.

The result of the lowering of the example to the Concrete Dialect is shown below:

```mlir
func.func @main(%arg0: tensor<4x3x513xi64>, %arg1: tensor<3x2xi3>) -> tensor<4x2x513xi64> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %generated = tensor.generate  {
  ^bb0(%arg2: index, %arg3: index, %arg4: index):
    %c0_i64 = arith.constant 0 : i64
    tensor.yield %c0_i64 : i64
  } : tensor<4x2x513xi64>
  %0 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %generated) -> (tensor<4x2x513xi64>) {
    %1 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %arg3) -> (tensor<4x2x513xi64>) {
      %2 = scf.for %arg6 = %c0 to %c3 step %c1 iter_args(%arg7 = %arg5) -> (tensor<4x2x513xi64>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg6, 0] [1, 1, 513] [1, 1, 1] : tensor<4x3x513xi64> to tensor<513xi64>
        %extracted = tensor.extract %arg1[%arg6, %arg4] : tensor<3x2xi3>
        %extracted_slice_0 = tensor.extract_slice %arg7[%arg2, %arg4, 0] [1, 1, 513] [1, 1, 1] : tensor<4x2x513xi64> to tensor<513xi64>
        %3 = arith.extsi %extracted : i3 to i64
        %4 = "Concrete.mul_cleartext_lwe_tensor"(%extracted_slice, %3) : (tensor<513xi64>, i64) -> tensor<513xi64>
        %5 = "Concrete.add_lwe_tensor"(%extracted_slice_0, %4) : (tensor<513xi64>, tensor<513xi64>) -> tensor<513xi64>
        %inserted_slice = tensor.insert_slice %5 into %arg7[%arg2, %arg4, 0] [1, 1, 513] [1, 1, 1] : tensor<513xi64> into tensor<4x2x513xi64>
        scf.yield %inserted_slice : tensor<4x2x513xi64>
      }
      scf.yield %2 : tensor<4x2x513xi64>
    }
    scf.yield %1 : tensor<4x2x513xi64>
  }
  return %0 : tensor<4x2x513xi64>
}
```

## Bufferization and emitting library calls

The remaining stages of the pipeline are rather technical. Before any binding to an actual Concrete backend library, the compiler first
invokes [MLIR's bufferization infrastructure](https://mlir.llvm.org/docs/Bufferization/) to convert the value-based IR into an IR with buffer semantics. In particular, this means that keys and encrypted values are no longer abstract values in a mathematical sense, but values backed by a memory location that holds the actual data. This form of IR is then suitable for a pass emitting actual library calls that implement the corresponding operations from the Concrete Dialect for a specific backend.

The result for the example is given below:

```mlir
func.func @main(%arg0: memref<4x3x513xi64, strided<[?, ?, ?], offset: ?>>, %arg1: memref<3x2xi3, strided<[?, ?], offset: ?>>, %arg2: !Concrete.context) -> memref<4x2x513xi64> {
  %c0_i64 = arith.constant 0 : i64
  call @_dfr_start(%c0_i64, %arg2) : (i64, !Concrete.context) -> ()
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c513 = arith.constant 513 : index
  %c0_i64_0 = arith.constant 0 : i64
  %c3 = arith.constant 3 : index
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x2x513xi64>
  scf.for %arg3 = %c0 to %c4 step %c1 {
    scf.for %arg4 = %c0 to %c2 step %c1 {
      scf.for %arg5 = %c0 to %c513 step %c1 {
        memref.store %c0_i64_0, %alloc[%arg3, %arg4, %arg5] : memref<4x2x513xi64>
      }
    }
  }
  scf.for %arg3 = %c0 to %c4 step %c1 {
    scf.for %arg4 = %c0 to %c2 step %c1 {
      %subview = memref.subview %alloc[%arg3, %arg4, 0] [1, 1, 513] [1, 1, 1] : memref<4x2x513xi64> to memref<513xi64, strided<[1], offset: ?>>
      scf.for %arg5 = %c0 to %c3 step %c1 {
        %subview_1 = memref.subview %arg0[%arg3, %arg5, 0] [1, 1, 513] [1, 1, 1] : memref<4x3x513xi64, strided<[?, ?, ?], offset: ?>> to memref<513xi64, strided<[?], offset: ?>>
        %0 = memref.load %arg1[%arg5, %arg4] : memref<3x2xi3, strided<[?, ?], offset: ?>>
        %1 = arith.extsi %0 : i3 to i64
        %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<513xi64>
        %cast = memref.cast %alloc_2 : memref<513xi64> to memref<?xi64, #map>
        %cast_3 = memref.cast %subview_1 : memref<513xi64, strided<[?], offset: ?>> to memref<?xi64, #map>
        func.call @memref_mul_cleartext_lwe_ciphertext_u64(%cast, %cast_3, %1) : (memref<?xi64, #map>, memref<?xi64, #map>, i64) -> ()
        %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<513xi64>
        %cast_5 = memref.cast %alloc_4 : memref<513xi64> to memref<?xi64, #map>
        %cast_6 = memref.cast %subview : memref<513xi64, strided<[1], offset: ?>> to memref<?xi64, #map>
        %cast_7 = memref.cast %alloc_2 : memref<513xi64> to memref<?xi64, #map>
        func.call @memref_add_lwe_ciphertexts_u64(%cast_5, %cast_6, %cast_7) : (memref<?xi64, #map>, memref<?xi64, #map>, memref<?xi64, #map>) -> ()
        memref.dealloc %alloc_2 : memref<513xi64>
        memref.copy %alloc_4, %subview : memref<513xi64> to memref<513xi64, strided<[1], offset: ?>>
        memref.dealloc %alloc_4 : memref<513xi64>
      }
    }
  }
  call @_dfr_stop(%c0_i64) : (i64) -> ()
  return %alloc : memref<4x2x513xi64>
}
```

At this stage, the IR is only composed of operations from builtin Dialects and thus amenable to lowering to LLVM-IR using the lowering passes provided by MLIR.
