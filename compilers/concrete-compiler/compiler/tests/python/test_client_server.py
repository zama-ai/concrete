import numpy as np
import pytest
import shutil
import tempfile

from concrete.compiler import (
    Library,
    Compiler,
    lookup_runtime_lib,
    CompilationOptions,
    Backend,
    Keyset,
    ClientProgram,
    ServerKeyset,
    Value,
    ServerProgram,
    TransportValue,
    CompilationContext,
)


@pytest.mark.parametrize(
    "mlir, args, expected_results",
    [
        pytest.param(
            """

func.func @main(%arg0: !FHE.eint<5>, %arg1: i6) -> !FHE.eint<5> {
    %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<5>, i6) -> (!FHE.eint<5>)
    return %1: !FHE.eint<5>
}

            """,
            (5, 7),
            (12,),
            id="enc_plain_int_args",
        ),
        pytest.param(
            """

func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
    %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.eint<5>)
    return %1: !FHE.eint<5>
}

            """,
            (5, 7),
            (12,),
            id="enc_enc_int_args",
        ),
        pytest.param(
            """

func.func @main(%arg0: tensor<4x!FHE.eint<5>>, %arg1: tensor<4xi6>) -> !FHE.eint<5> {
    %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) : (tensor<4x!FHE.eint<5>>, tensor<4xi6>) -> !FHE.eint<5>
    return %ret : !FHE.eint<5>
}

            """,
            (
                np.array([1, 2, 3, 4], dtype=np.int64),
                np.array([4, 3, 2, 1], dtype=np.int64),
            ),
            (20,),
            id="enc_plain_ndarray_args",
        ),
        pytest.param(
            """

func.func @main(%a0: tensor<4x!FHE.eint<5>>, %a1: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<5>>, tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>>
    return %res : tensor<4x!FHE.eint<5>>
}

            """,
            (
                np.array([1, 2, 3, 4], dtype=np.int64),
                np.array([7, 0, 1, 5], dtype=np.int64),
            ),
            (np.array([8, 2, 4, 9]),),
            id="enc_enc_ndarray_args",
        ),
        pytest.param(
            """
  func.func @main(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<4>, %arg2: !FHE.eint<5>, %arg3: !FHE.eint<6>) -> (!FHE.eint<3>, !FHE.eint<4>, !FHE.eint<5>, !FHE.eint<6>) {
    %lut0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %bs0 = "FHE.apply_lookup_table"(%arg0, %lut0): (!FHE.eint<3>, tensor<8xi64>) -> (!FHE.eint<3>)

    %lut1 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
    %bs1 = "FHE.apply_lookup_table"(%arg1, %lut1): (!FHE.eint<4>, tensor<16xi64>) -> (!FHE.eint<4>)

    %lut3 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>
    %bs3 = "FHE.apply_lookup_table"(%arg2, %lut3): (!FHE.eint<5>, tensor<32xi64>) -> (!FHE.eint<5>)

    %lut4 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
    %bs4 = "FHE.apply_lookup_table"(%arg3, %lut4): (!FHE.eint<6>, tensor<64xi64>) -> (!FHE.eint<6>)

    return %bs0, %bs1, %bs3, %bs4 : !FHE.eint<3>, !FHE.eint<4>, !FHE.eint<5>, !FHE.eint<6>
  }
            """,
            (7, 15, 31, 63),
            (7, 15, 31, 63),
            id="apply_lookup_table_multi_ouput",
        ),
    ],
)
def test_client_server_end_to_end(mlir, args, expected_results, keyset_cache):
    with tempfile.TemporaryDirectory() as tmpdirname:
        support = Compiler(
            str(tmpdirname), lookup_runtime_lib(), generate_shared_lib=True
        )
        library = support.compile(mlir, CompilationOptions(Backend.CPU))

        program_info = library.get_program_info()
        keyset = Keyset(program_info, keyset_cache)

        evaluation_keys = keyset.get_server_keys()
        evaluation_keys_serialized = evaluation_keys.serialize()
        evaluation_keys_deserialized = ServerKeyset.deserialize(
            evaluation_keys_serialized
        )

        client_program = ClientProgram.create_encrypted(program_info, keyset)
        client_circuit = client_program.get_client_circuit("main")
        args_serialized = [
            client_circuit.prepare_input(Value(arg), i).serialize()
            for (i, arg) in enumerate(args)
        ]
        args_deserialized = [TransportValue.deserialize(arg) for arg in args_serialized]

        server_program = ServerProgram(library, False)
        server_circuit = server_program.get_server_circuit("main")

        results = server_circuit.call(
            args_deserialized,
            evaluation_keys_deserialized,
        )
        results_serialized = [result.serialize() for result in results]
        results_deserialized = [
            client_circuit.process_output(
                TransportValue.deserialize(result), i
            ).to_py_val()
            for (i, result) in enumerate(results_serialized)
        ]

        assert all(
            [
                np.all(result == expected)
                for (result, expected) in zip(results_deserialized, expected_results)
            ]
        )
