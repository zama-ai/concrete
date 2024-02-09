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
from mlir._mlir_libs._concretelang._compiler import KeyType, PrimitiveOperation


def test_statistics():
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

        client_parameters = support.load_client_parameters(compilation_result)
        program_compilation_feedback = support.load_compilation_feedback(
            compilation_result
        )
        compilation_feedback = program_compilation_feedback.circuit("main")

        pbs_count = compilation_feedback.count(
            operations={
                PrimitiveOperation.PBS,
                PrimitiveOperation.WOP_PBS,
            }
        )
        assert pbs_count == 1

        pbs_counts_per_parameter = compilation_feedback.count_per_parameter(
            operations={
                PrimitiveOperation.PBS,
                PrimitiveOperation.WOP_PBS,
            },
            key_types={KeyType.BOOTSTRAP},
            client_parameters=client_parameters,
        )
        assert len(pbs_counts_per_parameter) == 1
        assert pbs_counts_per_parameter[list(pbs_counts_per_parameter.keys())[0]] == 1

        pbs_counts_per_tag = compilation_feedback.count_per_tag(
            operations={
                PrimitiveOperation.PBS,
                PrimitiveOperation.WOP_PBS,
            }
        )
        assert pbs_counts_per_tag == {}

        pbs_counts_per_tag_per_parameter = (
            compilation_feedback.count_per_tag_per_parameter(
                operations={
                    PrimitiveOperation.PBS,
                    PrimitiveOperation.WOP_PBS,
                },
                key_types={KeyType.BOOTSTRAP},
                client_parameters=client_parameters,
            )
        )
        assert pbs_counts_per_tag_per_parameter == {}
