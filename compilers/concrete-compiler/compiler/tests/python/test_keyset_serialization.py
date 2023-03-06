import numpy as np
import pytest
import shutil
import tempfile

from concrete.compiler import (
    ClientSupport,
    EvaluationKeys,
    KeySet,
    LibrarySupport,
    PublicArguments,
    PublicResult,
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
        support = LibrarySupport.new(str(tmpdirname))
        compilation_result = support.compile(mlir)

        server_lambda = support.load_server_lambda(compilation_result)
        client_parameters = support.load_client_parameters(compilation_result)

        keyset = ClientSupport.key_set(client_parameters)
        evaluation_keys = keyset.get_evaluation_keys()

        arg = 5
        encrypted_args = ClientSupport.encrypt_arguments(
            client_parameters, keyset, [arg]
        )

        result = support.server_call(server_lambda, encrypted_args, evaluation_keys)

        serialized_keyset = keyset.serialize()
        deserialized_keyset = KeySet.deserialize(serialized_keyset)

        output = ClientSupport.decrypt_result(
            client_parameters, deserialized_keyset, result
        )
        assert output == arg**2
