import numpy as np
import pytest
import shutil
import tempfile

from concrete.compiler import (
    Compiler,
    KeyType,
    PrimitiveOperation,
    lookup_runtime_lib,
    Backend,
    CompilationOptions,
    MoreCircuitCompilationFeedback,
)


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
        support = Compiler(str(tmpdirname), lookup_runtime_lib())
        library = support.compile(mlir, CompilationOptions(Backend.CPU))

        program_info = library.get_program_info()
        program_compilation_feedback = library.get_program_compilation_feedback()
        compilation_feedback = program_compilation_feedback.get_circuit_feedback("main")

        pbs_count = MoreCircuitCompilationFeedback.count(
            compilation_feedback,
            operations={
                PrimitiveOperation.PBS,
                PrimitiveOperation.WOP_PBS,
            },
        )
        assert pbs_count == 1

        pbs_counts_per_parameter = MoreCircuitCompilationFeedback.count_per_parameter(
            compilation_feedback,
            operations={
                PrimitiveOperation.PBS,
                PrimitiveOperation.WOP_PBS,
            },
            key_types={KeyType.BOOTSTRAP},
            program_info=program_info,
        )
        assert len(pbs_counts_per_parameter) == 1
        assert pbs_counts_per_parameter[list(pbs_counts_per_parameter.keys())[0]] == 1

        pbs_counts_per_tag = MoreCircuitCompilationFeedback.count_per_tag(
            compilation_feedback,
            operations={
                PrimitiveOperation.PBS,
                PrimitiveOperation.WOP_PBS,
            },
        )
        assert pbs_counts_per_tag == {}

        pbs_counts_per_tag_per_parameter = (
            MoreCircuitCompilationFeedback.count_per_tag_per_parameter(
                compilation_feedback,
                operations={
                    PrimitiveOperation.PBS,
                    PrimitiveOperation.WOP_PBS,
                },
                key_types={KeyType.BOOTSTRAP},
                program_info=program_info,
            )
        )
        assert pbs_counts_per_tag_per_parameter == {}
