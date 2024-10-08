import numpy as np
import pytest
import shutil
import tempfile

from concrete.compiler import (
    Compiler,
    CompilationOptions,
    lookup_runtime_lib,
    Keyset,
    ServerKeyset,
    ClientProgram,
    ServerProgram,
    TransportValue,
    Value,
    Backend,
)


def test_keyset_serialization():
    mlir = """

module {
  func.func @main(%arg0: !FHE.eint<3>) -> !FHE.eint<6> {
    %cst = arith.constant dense<[0, 1, 4, 9, 16, 25, 36, 49]> : tensor<8xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<6>
    return %0 : !FHE.eint<6>
  }
}

    """.strip()

    with tempfile.TemporaryDirectory() as tmpdirname:

        args = (5,)
        expected_results = (25,)

        support = Compiler(
            str(tmpdirname), lookup_runtime_lib(), generate_shared_lib=True
        )
        library = support.compile(mlir, CompilationOptions(Backend.CPU))

        program_info = library.get_program_info()
        keyset = Keyset(program_info, None)
        keyset = Keyset.deserialize(keyset.serialize())

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
