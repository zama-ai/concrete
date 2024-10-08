import platform
import pytest
import os.path
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


def run(library: Library, args, keyset_cache, circuit_name):
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
    keyset = Keyset(program_info, keyset_cache)

    evaluation_keys = keyset.get_server_keys()
    evaluation_keys_serialized = evaluation_keys.serialize()
    evaluation_keys_deserialized = ServerKeyset.deserialize(evaluation_keys_serialized)

    client_program = ClientProgram.create_encrypted(program_info, keyset)
    client_circuit = client_program.get_client_circuit(circuit_name)
    args_serialized = [
        client_circuit.prepare_input(Value(arg), i).serialize()
        for (i, arg) in enumerate(args)
    ]
    args_deserialized = [TransportValue.deserialize(arg) for arg in args_serialized]

    server_program = ServerProgram(library, False)
    server_circuit = server_program.get_server_circuit(circuit_name)

    results = server_circuit.call(
        args_deserialized,
        evaluation_keys_deserialized,
    )
    results_serialized = [result.serialize() for result in results]
    results_deserialized = [
        client_circuit.process_output(TransportValue.deserialize(result), i).to_py_val()
        for (i, result) in enumerate(results_serialized)
    ]

    return results_deserialized


def compile_run_assert(
    compiler,
    mlir_input,
    args,
    expected_result,
    keyset_cache,
    options=CompilationOptions(Backend.CPU),
    circuit_name="main",
):
    """Compile run and assert result."""
    library = compiler.compile(mlir_input, options)
    result = run(library, args, keyset_cache, circuit_name)
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
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.array(4), np.array(5)),
        (9,),
        id="add_eint_int_with_ndarray_as_scalar",
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
        (
            np.array(
                [-5, 3],
            ),
        ),
        (np.array([5, -3]),),
        id="neg_eint_signed_2",
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
def test_lib_compile_and_run(mlir_input, args, expected_result, keyset_cache):
    artifact_dir = "./py_test_lib_compile_and_run"
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    compile_run_assert(compiler, mlir_input, args, expected_result, keyset_cache)
    shutil.rmtree(artifact_dir)


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_reload_and_run(mlir_input, args, expected_result, keyset_cache):
    artifact_dir = "./test_lib_compile_reload_and_run"
    library = Library(artifact_dir)
    # Here don't save compilation result, reload
    library = Compiler(artifact_dir, lookup_runtime_lib()).compile(
        mlir_input, CompilationOptions(Backend.CPU)
    )
    compilation_result = library.get_program_compilation_feedback()
    result = run(library, args, keyset_cache, "main")
    # Check result
    assert_result(result, expected_result)
    shutil.rmtree(artifact_dir)


def test_lib_compilation_artifacts():
    mlir_str = """
    func.func @main(%a0: tensor<4x!FHE.eint<6>>, %a1: tensor<4xi7>) -> tensor<4x!FHE.eint<6>> {
                %res = "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<6>>, tensor<4xi7>) -> tensor<4x!FHE.eint<6>>
                return %res : tensor<4x!FHE.eint<6>>
    }
    """
    artifact_dir = "./test_artifacts"
    library = Compiler(artifact_dir, lookup_runtime_lib()).compile(
        mlir_str, CompilationOptions(Backend.CPU)
    )
    assert os.path.exists(library.get_program_info_path())
    assert os.path.exists(library.get_shared_lib_path())
    shutil.rmtree(artifact_dir)
    assert not os.path.exists(library.get_program_info_path())
    assert not os.path.exists(library.get_shared_lib_path())


def test_multi_circuits(keyset_cache):
    from mlir._mlir_libs._concretelang._compiler import OptimizerStrategy

    mlir_str = """
    func.func @add(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
        %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
        return %1: !FHE.eint<7>
    }
    func.func @sub(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
        %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
        return %1: !FHE.eint<7>
    }
    """
    args = (10, 3)
    expected_add_result = (13,)
    expected_sub_result = (7,)
    artifact_dir = "./py_test_multi_circuits"
    options = CompilationOptions(Backend.CPU)
    options.set_optimizer_strategy(OptimizerStrategy.V0)
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    compile_run_assert(
        compiler,
        mlir_str,
        args,
        expected_add_result,
        keyset_cache,
        options,
        circuit_name="add",
    )
    compile_run_assert(
        compiler,
        mlir_str,
        args,
        expected_sub_result,
        keyset_cache,
        options,
        circuit_name="sub",
    )


def _test_lib_compile_and_run_with_options(keyset_cache, options):
    mlir_input = """
        func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
            %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
            %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
            return %1: !FHE.eint<7>
        }
    """
    args = (73,)
    expected_result = (73,)
    compiler = Compiler(
        "./py_test_lib_compile_and_run_custom_perror", lookup_runtime_lib()
    )

    compile_run_assert(
        compiler, mlir_input, args, expected_result, keyset_cache, options
    )


def test_lib_compile_and_run_p_error(keyset_cache):
    options = CompilationOptions(Backend.CPU)
    options.set_p_error(0.00001)
    options.set_display_optimizer_choice(True)
    _test_lib_compile_and_run_with_options(keyset_cache, options)


def test_lib_compile_and_run_global_p_error(keyset_cache):
    options = CompilationOptions(Backend.CPU)
    options.set_global_p_error(0.00001)
    options.set_display_optimizer_choice(True)
    _test_lib_compile_and_run_with_options(keyset_cache, options)


@pytest.mark.parallel
@pytest.mark.parametrize(
    "mlir_input, args, expected_result", end_to_end_parallel_fixture
)
def test_compile_and_run_auto_parallelize(
    mlir_input, args, expected_result, keyset_cache
):
    artifact_dir = "./py_test_compile_and_run_auto_parallelize"
    options = CompilationOptions(Backend.CPU)
    options.set_auto_parallelize(True)
    engine = Compiler(artifact_dir, lookup_runtime_lib())
    compile_run_assert(
        engine, mlir_input, args, expected_result, keyset_cache, options=options
    )


@pytest.mark.parametrize(
    "mlir_input, args, expected_result",
    [
        pytest.param(
            """
            func.func @main(%x: tensor<3x4x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
                %y = arith.constant dense<[[1, 2], [3, 4], [5, 0], [1, 2]]> : tensor<4x2xi8>
                %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x4x!FHE.eint<7>>, tensor<4x2xi8>) -> tensor<3x2x!FHE.eint<7>>
                return %0 : tensor<3x2x!FHE.eint<7>>
            }
            """,
            (np.array([[1, 2, 3, 4], [4, 2, 1, 0], [2, 3, 1, 5]]),),
            (np.array([[26, 18], [15, 16], [21, 26]]),),
            id="matmul_eint_int_uint8",
        ),
    ],
)
def test_compile_and_run_loop_parallelize(
    mlir_input, args, expected_result, keyset_cache
):
    artifact_dir = "./py_test_compile_and_run_loop_parallelize"
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    options = CompilationOptions(Backend.CPU)
    options.set_loop_parallelize(True)
    compile_run_assert(
        compiler, mlir_input, args, expected_result, keyset_cache, options=options
    )


@pytest.mark.parametrize(
    "mlir_input, args",
    [
        pytest.param(
            """
            func.func @main%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
            (5, 7, 8),
            id="add_eint_int_invalid_arg_number",
        ),
    ],
)
def test_compile_and_run_invalid_arg_number(mlir_input, args, keyset_cache):
    artifact_dir = "./py_test_compile_and_run_invalid_arg_number"
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    with pytest.raises(RuntimeError):
        compile_run_assert(compiler, mlir_input, args, None, keyset_cache)


def test_crt_decomposition_feedback():
    mlir = """

func.func @main(%arg0: !FHE.eint<16>) -> !FHE.eint<16> {
    %tlu = arith.constant dense<60000> : tensor<65536xi64>
    %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<16>, tensor<65536xi64>) -> (!FHE.eint<16>)
    return %1: !FHE.eint<16>
}

    """

    artifact_dir = "./py_test_crt_decomposition_feedback"
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    library = compiler.compile(mlir, options=CompilationOptions(Backend.CPU))
    compilation_feedback = library.get_program_compilation_feedback()

    assert isinstance(compilation_feedback, ProgramCompilationFeedback)
    assert isinstance(compilation_feedback.complexity, float)
    assert isinstance(compilation_feedback.p_error, float)
    assert isinstance(compilation_feedback.global_p_error, float)
    assert isinstance(compilation_feedback.total_secret_keys_size, int)
    assert isinstance(compilation_feedback.total_bootstrap_keys_size, int)
    assert isinstance(compilation_feedback.circuit_feedbacks, list)
    assert isinstance(
        compilation_feedback.circuit_feedbacks[0], CircuitCompilationFeedback
    )
    assert isinstance(compilation_feedback.circuit_feedbacks[0].total_inputs_size, int)
    assert isinstance(compilation_feedback.circuit_feedbacks[0].total_output_size, int)
    assert isinstance(
        compilation_feedback.circuit_feedbacks[0].crt_decompositions_of_outputs, list
    )
    assert compilation_feedback.circuit_feedbacks[0].crt_decompositions_of_outputs == [
        [7, 8, 9, 11, 13]
    ]


@pytest.mark.parametrize(
    "mlir, expected_memory_usage_per_loc",
    [
        pytest.param(
            """
            func.func @main(%arg0: tensor<4x4x!FHE.eint<6>>, %arg1: tensor<4x2xi7>) -> tensor<4x2x!FHE.eint<6>> {
                %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<4x4x!FHE.eint<6>>, tensor<4x2xi7>) -> (tensor<4x2x!FHE.eint<6>>) loc("some/random/location.py":10:2)
                %tlu = arith.constant dense<[40, 13, 20, 62, 47, 41, 46, 30, 59, 58, 17, 4, 34, 44, 49, 5, 10, 63, 18, 21, 33, 45, 7, 14, 24, 53, 56, 3, 22, 29, 1, 39, 48, 32, 38, 28, 15, 12, 52, 35, 42, 11, 6, 43, 0, 16, 27, 9, 31, 51, 36, 37, 55, 57, 54, 2, 8, 25, 50, 23, 61, 60, 26, 19]> : tensor<64xi64> loc("some/random/location.py":10:2)
                %result = "FHELinalg.apply_lookup_table"(%0, %tlu): (tensor<4x2x!FHE.eint<6>>, tensor<64xi64>) -> (tensor<4x2x!FHE.eint<6>>) loc("some/random/location.py":10:2)
                return %result: tensor<4x2x!FHE.eint<6>> loc("some/random/location.py":10:2)
            }
            """,
            # 4*4*4097*8 (input1) + 4*2 (input2) + 4*2*4097*8 + 4097*3*8 + 4096*8 + 869*8 (temporary buffers) + 4*2*4097*8 (output buffer) + 64*8 (constant TLU)
            {'loc("some/random/location.py":10:2)': 1187584},
            id="single location",
        ),
        pytest.param(
            """
            func.func @main(%arg0: tensor<4x4x!FHE.eint<6>>, %arg1: tensor<4x2xi7>) -> tensor<4x2x!FHE.eint<6>> {
                %0 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<4x4x!FHE.eint<6>>, tensor<4x2xi7>) -> (tensor<4x2x!FHE.eint<6>>) loc("@matmul some/random/location.py":10:2)
                %tlu = arith.constant dense<[40, 13, 20, 62, 47, 41, 46, 30, 59, 58, 17, 4, 34, 44, 49, 5, 10, 63, 18, 21, 33, 45, 7, 14, 24, 53, 56, 3, 22, 29, 1, 39, 48, 32, 38, 28, 15, 12, 52, 35, 42, 11, 6, 43, 0, 16, 27, 9, 31, 51, 36, 37, 55, 57, 54, 2, 8, 25, 50, 23, 61, 60, 26, 19]> : tensor<64xi64> loc("@lut some/random/location.py":11:2)
                %result = "FHELinalg.apply_lookup_table"(%0, %tlu): (tensor<4x2x!FHE.eint<6>>, tensor<64xi64>) -> (tensor<4x2x!FHE.eint<6>>) loc("@lut some/random/location.py":11:2)
                return %result: tensor<4x2x!FHE.eint<6>> loc("@return some/random/location.py":12:2)
            }
            """,
            {
                # 4*4*4097*8 (input1) + 4*2 (input2) + 4*2*4097*8 (matmul result buffer) + 4097*2*8 (temporary buffers)
                'loc("@matmul some/random/location.py":10:2)': 852184,
                # 4*2*4097*8 (matmul result buffer) + 4*2*4097*8 (result buffer) + 4097*8 + 4096*8 + 869*8 (temporary buffers) + 64*8 (constant TLU)
                'loc("@lut some/random/location.py":11:2)': 597608,
                # 4*2*4097*8 (result buffer)
                'loc("@return some/random/location.py":12:2)': 262208,
            },
            id="multiple location",
        ),
    ],
)
def test_memory_usage(mlir: str, expected_memory_usage_per_loc: dict):
    artifact_dir = "./test_memory_usage"
    compiler = Compiler(artifact_dir, lookup_runtime_lib())
    library = compiler.compile(mlir, CompilationOptions(Backend.CPU))
    compilation_feedback = library.get_program_compilation_feedback()
    assert isinstance(compilation_feedback, ProgramCompilationFeedback)

    assert (
        expected_memory_usage_per_loc
        == compilation_feedback.circuit_feedbacks[0].memory_usage_per_location
    )

    shutil.rmtree(artifact_dir)
