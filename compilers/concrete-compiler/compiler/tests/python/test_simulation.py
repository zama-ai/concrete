import json
import subprocess
import sys
import os
import tempfile
import pytest
import shutil
import numpy as np
from concrete.compiler import (
    Compiler,
    CompilationOptions,
    ProgramCompilationFeedback,
    CircuitCompilationFeedback,
    Backend,
    lookup_runtime_lib,
    Keyset,
    Library,
    ServerKeyset,
    ServerProgram,
    ClientProgram,
    TransportValue,
    Value,
)


def assert_result(results, expected_results):
    """Assert that result and expected result are equal.

    result and expected_result can be integers on numpy arrays.
    """
    for result, expected_result in zip(results, expected_results):
        assert type(expected_result) == type(result)
        assert np.all(result == expected_result)


def run_simulated(library: Library, args, circuit_name):
    """Execute engine on the given arguments.

    Perform required loading, encryption, execution, and decryption."""
    # Dev
    compilation_feedback = library.get_program_compilation_feedback()
    assert isinstance(compilation_feedback, ProgramCompilationFeedback)
    assert isinstance(compilation_feedback.complexity, float)
    assert isinstance(compilation_feedback.p_error, float)
    assert isinstance(compilation_feedback.global_p_error, float)
    assert isinstance(compilation_feedback.total_secret_keys_size, int)
    assert isinstance(compilation_feedback.total_bootstrap_keys_size, int)
    assert isinstance(compilation_feedback.circuit_feedbacks, list)
    circuit_feedback = compilation_feedback.get_circuit_feedback(circuit_name)
    assert isinstance(circuit_feedback, CircuitCompilationFeedback)
    assert isinstance(circuit_feedback.total_inputs_size, int)
    assert isinstance(circuit_feedback.total_output_size, int)

    program_info = library.get_program_info()

    client_program = ClientProgram.create_simulated(program_info)
    client_circuit = client_program.get_client_circuit(circuit_name)
    args_serialized = [
        client_circuit.simulate_prepare_input(Value(arg), i).serialize()
        for (i, arg) in enumerate(args)
    ]
    args_deserialized = [TransportValue.deserialize(arg) for arg in args_serialized]

    server_program = ServerProgram(library, True)
    server_circuit = server_program.get_server_circuit(circuit_name)

    results = server_circuit.simulate(args_deserialized)
    results_serialized = [result.serialize() for result in results]
    results_deserialized = [
        client_circuit.simulate_process_output(
            TransportValue.deserialize(result), i
        ).to_py_val()
        for (i, result) in enumerate(results_serialized)
    ]

    return results_deserialized


def compile_run_assert(
    compiler,
    mlir_input,
    args,
    expected_result,
    options=CompilationOptions(Backend.CPU),
    circuit_name="main",
):
    """Compile run and assert result."""
    options.simulation(True)
    options.set_enable_overflow_detection_in_simulation(True)
    library = compiler.compile(mlir_input, options)
    result = run_simulated(library, args, circuit_name)
    assert_result(result, expected_result)


end_to_end_fixture = [
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (5, 7),
        (12,),
        id="add_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
                %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (73,),
        (73,),
        id="apply_lookup_table",
    ),
    pytest.param(
        """
            func.func @main(%arg0: tensor<4x!FHE.eint<7>>, %arg1: tensor<4xi8>) -> !FHE.eint<7>
            {
                %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!FHE.eint<7>>, tensor<4xi8>) -> !FHE.eint<7>
                return %ret : !FHE.eint<7>
            }
            """,
        (
            np.array([1, 2, 3, 4]),
            np.array([4, 3, 2, 1]),
        ),
        (20,),
        id="dot_eint_int_uint8",
    ),
    pytest.param(
        """
            func.func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
                %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
                return %res : tensor<4x!FHE.eint<6>>
            }
            """,
        (
            np.array([31, 6, 12, 9]),
            np.array([32, 9, 2, 3]),
        ),
        (np.array([63, 15, 14, 12]),),
        id="add_eint_int_1D",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
                %0 = "FHE.neg_eint"(%arg0): (!FHE.esint<7>) -> !FHE.esint<7>
                return %0: !FHE.esint<7>
            }
            """,
        (5,),
        (-5,),
        id="neg_eint_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: tensor<2x!FHE.esint<7>>) -> tensor<2x!FHE.esint<7>> {
                %0 = "FHELinalg.neg_eint"(%arg0): (tensor<2x!FHE.esint<7>>) -> tensor<2x!FHE.esint<7>>
                return %0: tensor<2x!FHE.esint<7>>
            }
            """,
        (np.array([-5, 3]),),
        (np.array([5, -3]),),
        id="neg_eint_signed_2",
    ),
    pytest.param(
        """
        func.func @main(%arg0: tensor<4x4x!FHE.eint<13>>, %arg1: tensor<4x4xi14>) -> tensor<4x4x!FHE.eint<13>> {
            %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1) : (tensor<4x4x!FHE.eint<13>>, tensor<4x4xi14>) -> tensor<4x4x!FHE.eint<13>>
            %1 = "FHELinalg.matmul_eint_int"(%0, %arg1) : (tensor<4x4x!FHE.eint<13>>, tensor<4x4xi14>) -> tensor<4x4x!FHE.eint<13>>
            %2 = "FHELinalg.matmul_eint_int"(%1, %arg1) : (tensor<4x4x!FHE.eint<13>>, tensor<4x4xi14>) -> tensor<4x4x!FHE.eint<13>>
            %3 = "FHELinalg.matmul_eint_int"(%2, %arg1) : (tensor<4x4x!FHE.eint<13>>, tensor<4x4xi14>) -> tensor<4x4x!FHE.eint<13>>
            return %3 : tensor<4x4x!FHE.eint<13>>
        }
        """,
        (
            np.array([i // 4 for i in range(16)]).reshape((4, 4)),
            np.array([i // 4 for i in range(15, -1, -1)]).reshape((4, 4)),
        ),
        (
            np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    1296,
                    1296,
                    1296,
                    1296,
                    2592,
                    2592,
                    2592,
                    2592,
                    3888,
                    3888,
                    3888,
                    3888,
                ]
            ).reshape((4, 4)),
        ),
        id="matul_chain_with_crt",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<14>, %arg1: tensor<16384xi64>) -> !FHE.eint<14> {
                %cst = arith.constant 15 : i15
                %v = "FHE.add_eint_int"(%arg0, %cst): (!FHE.eint<14>, i15) -> (!FHE.eint<14>)
                %1 = "FHE.apply_lookup_table"(%v, %arg1): (!FHE.eint<14>, tensor<16384xi64>) -> (!FHE.eint<14>)
                return %1: !FHE.eint<14>
            }
        """,
        (
            81,
            np.array(range(16384)),
        ),
        (96,),
        id="add_lut_crt",
    ),
]

end_to_end_parallel_fixture = [
    pytest.param(
        """
            func.func @main(%x: tensor<3x4x!FHE.eint<7>>, %y: tensor<3x4x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
                %c = arith.constant dense<[[1, 2], [3, 4], [5, 0], [1, 2]]> : tensor<4x2xi8>
                %0 = "FHELinalg.matmul_eint_int"(%x, %c): (tensor<3x4x!FHE.eint<7>>, tensor<4x2xi8>) -> tensor<3x2x!FHE.eint<7>>
                %1 = "FHELinalg.matmul_eint_int"(%y, %c): (tensor<3x4x!FHE.eint<7>>, tensor<4x2xi8>) -> tensor<3x2x!FHE.eint<7>>
                %2 = "FHELinalg.add_eint"(%0, %1): (tensor<3x2x!FHE.eint<7>>, tensor<3x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>
                return %2 : tensor<3x2x!FHE.eint<7>>
            }
            """,
        (
            np.array([[1, 2, 3, 4], [4, 2, 1, 0], [2, 3, 1, 5]]),
            np.array([[1, 2, 3, 4], [4, 2, 1, 1], [2, 3, 1, 5]]),
        ),
        (np.array([[52, 36], [31, 34], [42, 52]]),),
        id="matmul_eint_int_uint8",
    ),
    pytest.param(
        """
            func.func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>, %a2: tensor<4x!FHE.eint<6>>, %a3: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
                %1 = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
                %2 = "FHELinalg.add_eint_int"(%a2, %a3) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
                %res = "FHELinalg.add_eint"(%1, %2) : (tensor<4x!FHE.eint<6>>, tensor<4x!FHE.eint<6>>) -> tensor<4x!FHE.eint<6>>
                return %res : tensor<4x!FHE.eint<6>>
            }
            """,
        (
            np.array([1, 2, 3, 4]),
            np.array([9, 8, 6, 5]),
            np.array([3, 2, 7, 0]),
            np.array([1, 4, 2, 11]),
        ),
        (np.array([14, 16, 18, 20]),),
        id="add_eint_int_1D",
    ),
]


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_and_run_simulation(mlir_input, args, expected_result):
    artifact_dir = "./py_test_lib_compile_and_run"
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    compile_run_assert(compiler, mlir_input, args, expected_result)
    shutil.rmtree(artifact_dir)


end_to_end_overflow_simu_fixture = [
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (120, 30),
        (150,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-1, -2),
        (-3,),
        b"",
        id="add_eint_int_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-60, -20),
        (-80,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint_int_signed_underflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (60, 20),
        (-48,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint_int_signed_overflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
                %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (81, 73),
        (154,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: !FHE.esint<7>) -> !FHE.esint<7> {
                %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.esint<7>, !FHE.esint<7>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-81, 73),
        (-8,),
        b"",
        id="add_eint_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: !FHE.esint<7>) -> !FHE.esint<7> {
                %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.esint<7>, !FHE.esint<7>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-60, -20),
        (-80,),  # undefined behavior
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint_signed_underflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: !FHE.esint<7>) -> !FHE.esint<7> {
                %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.esint<7>, !FHE.esint<7>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (81, 73),
        (-102,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint_signed_overflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.sub_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (4, 7),
        (256 - 3,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="sub_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.sub_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (4, 7),
        (-3,),
        b"",
        id="sub_eint_int_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.sub_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-37, 40),
        (-77,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="sub_eint_int_signed_underflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.sub_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (33, -40),
        (-55,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="sub_eint_int_signed_overflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
                %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (11, 18),
        (256 - 7,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="sub_eint",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: !FHE.esint<7>) -> !FHE.esint<7> {
                %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.esint<7>, !FHE.esint<7>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (11, 18),
        (-7,),
        b"",
        id="sub_eint_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: !FHE.esint<7>) -> !FHE.esint<7> {
                %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.esint<7>, !FHE.esint<7>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-44, 32),
        (-76,),  # undefined behavior
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="sub_eint_signed_underflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: !FHE.esint<7>) -> !FHE.esint<7> {
                %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.esint<7>, !FHE.esint<7>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (61, -25),
        (-42,),  # undefined behavior
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="sub_eint_signed_overflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (20, 10),
        (200,),
        b'WARNING at loc("-":3:22): overflow happened during multiplication in simulation\n',
        id="mul_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.sub_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                %2 = "FHE.mul_eint_int"(%1, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %2: !FHE.eint<7>
            }
            """,
        (5, 10),
        (256 - 50,),
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\nWARNING at loc("-":4:22): overflow happened during multiplication in simulation\n',
        id="sub_mul_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (5, -2),
        (-10,),
        b"",
        id="mul_eint_int_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-33, 5),
        (-37,),  # undefined behavior
        b'WARNING at loc("-":3:22): overflow happened during multiplication in simulation\n',
        id="mul_eint_int_signed_underflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>, %arg1: i8) -> !FHE.esint<7> {
                %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.esint<7>, i8) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-33, -5),
        (-91,),
        b'WARNING at loc("-":3:22): overflow happened during multiplication in simulation\n',
        id="mul_eint_int_signed_overflow",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
                %tlu = arith.constant dense<[0, 1420, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (1,),
        (140,),
        b'WARNING at loc("-":4:22): overflow (padding bit) happened during LUT in simulation\nWARNING at loc("-":4:22): overflow (original value didn\'t fit, so a modulus was applied) happened during LUT in simulation\n',
        id="apply_lookup_table_big_value",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
                %tlu = arith.constant dense<[0, 1400, 254, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.esint<7>, tensor<128xi64>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (2,),
        (-2,),
        b"",
        id="apply_lookup_table_signed",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
                %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 12]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.esint<7>, tensor<128xi64>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (-1,),
        (12,),
        b"",
        id="apply_lookup_table_signed_with_negative",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.esint<7>) -> !FHE.esint<7> {
                %tlu = arith.constant dense<[0, 1400, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.esint<7>, tensor<128xi64>) -> (!FHE.esint<7>)
                return %1: !FHE.esint<7>
            }
            """,
        (1,),
        (-8,),
        b'WARNING at loc("-":4:22): overflow (original value didn\'t fit, so a modulus was applied) happened during LUT in simulation\n',
        id="apply_lookup_table_signed_big_value",
    ),
]


@pytest.mark.parametrize(
    "mlir_input, args, expected_result, overflow_message",
    end_to_end_overflow_simu_fixture,
)
def test_lib_compile_and_run_simulation_with_overflow(
    mlir_input, args, expected_result, overflow_message
):
    # write mlir to tmp file
    mlir_file = tempfile.NamedTemporaryFile("w")
    mlir_file.write(mlir_input)
    mlir_file.flush()

    # prepare cmd and run
    script_path = os.path.join(os.path.dirname(__file__), "overflow.py")
    cmd = [sys.executable, script_path, mlir_file.name]
    cmd.append(json.dumps((args, expected_result)))
    out = subprocess.check_output(cmd, env=os.environ)

    # close/remove tmp file
    mlir_file.close()

    assert overflow_message == out
