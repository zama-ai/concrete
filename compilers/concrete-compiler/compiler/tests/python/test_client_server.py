import numpy as np
import pytest
import shutil
import tempfile

from concrete.compiler import (
    ClientSupport,
    EvaluationKeys,
    LibrarySupport,
    PublicArguments,
    PublicResult,
)


@pytest.mark.parametrize(
    "mlir, args, expected_result",
    [
        pytest.param(
            """

func.func @main(%arg0: !FHE.eint<5>, %arg1: i6) -> !FHE.eint<5> {
    %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<5>, i6) -> (!FHE.eint<5>)
    return %1: !FHE.eint<5>
}

            """,
            (5, 7),
            12,
            id="enc_plain_int_args",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            """

func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
    %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<5>, !FHE.eint<5>) -> (!FHE.eint<5>)
    return %1: !FHE.eint<5>
}

            """,
            (5, 7),
            12,
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
                np.array([1, 2, 3, 4], dtype=np.uint64),
                np.array([4, 3, 2, 1], dtype=np.uint8),
            ),
            20,
            id="enc_plain_ndarray_args",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            """

func.func @main(%a0: tensor<4x!FHE.eint<5>>, %a1: tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>> {
    %res = "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<5>>, tensor<4x!FHE.eint<5>>) -> tensor<4x!FHE.eint<5>>
    return %res : tensor<4x!FHE.eint<5>>
}

            """,
            (
                np.array([1, 2, 3, 4], dtype=np.uint64),
                np.array([7, 0, 1, 5], dtype=np.uint64),
            ),
            np.array([8, 2, 4, 9]),
            id="enc_enc_ndarray_args",
        ),
    ],
)
def test_client_server_end_to_end(mlir, args, expected_result, keyset_cache):
    with tempfile.TemporaryDirectory() as tmpdirname:
        support = LibrarySupport.new(str(tmpdirname))
        compilation_result = support.compile(mlir)
        server_lambda = support.load_server_lambda(compilation_result)

        client_parameters = support.load_client_parameters(compilation_result)
        keyset = ClientSupport.key_set(client_parameters, keyset_cache)

        evaluation_keys = keyset.get_evaluation_keys()
        evaluation_keys_serialized = evaluation_keys.serialize()
        evaluation_keys_deserialized = EvaluationKeys.deserialize(
            evaluation_keys_serialized
        )

        args = ClientSupport.encrypt_arguments(client_parameters, keyset, args)
        args_serialized = args.serialize()
        args_deserialized = PublicArguments.deserialize(
            client_parameters, args_serialized
        )

        result = support.server_call(
            server_lambda,
            args_deserialized,
            evaluation_keys_deserialized,
        )
        result_serialized = result.serialize()
        result_deserialized = PublicResult.deserialize(
            client_parameters, result_serialized
        )

        output = ClientSupport.decrypt_result(
            client_parameters, keyset, result_deserialized
        )
        assert np.array_equal(output, expected_result)
