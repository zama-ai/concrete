import subprocess
import sys
import os
import tempfile
import pytest
import shutil
import numpy as np
from concrete.compiler import (
    LibrarySupport,
    PublicArguments,
    SimulatedValueExporter,
    SimulatedValueDecrypter,
    CompilationOptions,
)


def assert_result(result, expected_result):
    """Assert that result and expected result are equal.

    result and expected_result can be integers on numpy arrays.
    """
    assert type(expected_result) == type(result)
    if isinstance(expected_result, int):
        assert result == expected_result
    else:
        assert np.all(result == expected_result)


def run_simulated(engine, args_and_shape, compilation_result):
    client_parameters = engine.load_client_parameters(compilation_result)
    sim_value_exporter = SimulatedValueExporter.new(client_parameters)
    values = []
    pos = 0
    for arg, shape in args_and_shape:
        if shape is None:
            assert isinstance(arg, int)
            values.append(sim_value_exporter.export_scalar(pos, arg))
        else:
            assert isinstance(arg, list)
            assert isinstance(shape, list)
            values.append(sim_value_exporter.export_tensor(pos, arg, shape))
        pos += 1
    public_arguments = PublicArguments.new(client_parameters, values)
    server_lambda = engine.load_server_lambda(compilation_result, True)
    public_result = engine.simulate(server_lambda, public_arguments)
    sim_value_decrypter = SimulatedValueDecrypter.new(client_parameters)
    result = sim_value_decrypter.decrypt(0, public_result.get_value(0))
    return result


def compile_run_assert(
    engine,
    mlir_input,
    args_and_shape,
    expected_result,
    options=CompilationOptions.new(),
):
    # compile with simulation
    options.simulation(True)
    compilation_result = engine.compile(mlir_input, options)
    result = run_simulated(engine, args_and_shape, compilation_result)
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
        12,
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
        73,
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
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([4, 3, 2, 1], dtype=np.uint8),
        ),
        20,
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
            np.array([31, 6, 12, 9], dtype=np.uint8),
            np.array([32, 9, 2, 3], dtype=np.uint8),
        ),
        np.array([63, 15, 14, 12]),
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
        -5,
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
        np.array([5, -3]),
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
            np.array(range(16384), dtype=np.uint64),
        ),
        96,
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
            np.array([[1, 2, 3, 4], [4, 2, 1, 0], [2, 3, 1, 5]], dtype=np.uint8),
            np.array([[1, 2, 3, 4], [4, 2, 1, 1], [2, 3, 1, 5]], dtype=np.uint8),
        ),
        np.array([[52, 36], [31, 34], [42, 52]]),
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
            np.array([1, 2, 3, 4], dtype=np.uint8),
            np.array([9, 8, 6, 5], dtype=np.uint8),
            np.array([3, 2, 7, 0], dtype=np.uint8),
            np.array([1, 4, 2, 11], dtype=np.uint8),
        ),
        np.array([14, 16, 18, 20]),
        id="add_eint_int_1D",
    ),
]


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_and_run_simulation(mlir_input, args, expected_result):
    artifact_dir = "./py_test_lib_compile_and_run"
    engine = LibrarySupport.new(artifact_dir)
    args_and_shape = []
    for arg in args:
        if isinstance(arg, int):
            args_and_shape.append((arg, None))
        else:  # np.array
            args_and_shape.append((arg.flatten().tolist(), list(arg.shape)))
    compile_run_assert(engine, mlir_input, args_and_shape, expected_result)
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
        150,
        b'WARNING at loc("-":3:22): overflow happened during addition in simulation\n',
        id="add_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.mul_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (20, 10),
        200,
        b'WARNING at loc("-":3:22): overflow happened during multiplication in simulation\n',
        id="mul_eint_int",
    ),
    pytest.param(
        """
            func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
                %tlu = arith.constant dense<[0, 140, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
                %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (1,),
        140,
        b'WARNING at loc("-":4:22): overflow happened during LUT in simulation\n',
        id="apply_lookup_table",
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
    cmd.extend(map(str, args))
    cmd.append(str(expected_result))
    out = subprocess.check_output(cmd, env=os.environ)

    # close/remove tmp file
    mlir_file.close()

    assert overflow_message == out
