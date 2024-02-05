import platform
import pytest
import os.path
import shutil
import numpy as np
from concrete.compiler import (
    LibrarySupport,
    ClientSupport,
    CompilationOptions,
    CompilationFeedback,
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


def run(engine, args, compilation_result, keyset_cache):
    """Execute engine on the given arguments.

    Perform required loading, encryption, execution, and decryption."""
    # Dev
    compilation_feedback = engine.load_compilation_feedback(compilation_result)
    assert isinstance(compilation_feedback, CompilationFeedback)
    assert isinstance(compilation_feedback.complexity, float)
    assert isinstance(compilation_feedback.p_error, float)
    assert isinstance(compilation_feedback.global_p_error, float)
    assert isinstance(compilation_feedback.total_secret_keys_size, int)
    assert isinstance(compilation_feedback.total_bootstrap_keys_size, int)
    assert isinstance(compilation_feedback.total_inputs_size, int)
    assert isinstance(compilation_feedback.total_output_size, int)

    # Client
    client_parameters = engine.load_client_parameters(compilation_result)
    key_set = ClientSupport.key_set(client_parameters, keyset_cache)
    public_arguments = ClientSupport.encrypt_arguments(client_parameters, key_set, args)
    # Server
    server_lambda = engine.load_server_lambda(compilation_result, False)
    evaluation_keys = key_set.get_evaluation_keys()
    public_result = engine.server_call(server_lambda, public_arguments, evaluation_keys)
    # Client
    result = ClientSupport.decrypt_result(client_parameters, key_set, public_result)
    return result


def compile_run_assert(
    engine,
    mlir_input,
    args,
    expected_result,
    keyset_cache,
    options=CompilationOptions.new("main"),
):
    """Compile run and assert result."""
    compilation_result = engine.compile(mlir_input, options)
    result = run(engine, args, compilation_result, keyset_cache)
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
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
                %1 = "FHE.add_eint_int"(%arg0, %arg1): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
                return %1: !FHE.eint<7>
            }
            """,
        (np.array(4, dtype=np.int64), np.array(5, dtype=np.uint8)),
        9,
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
def test_lib_compile_and_run(mlir_input, args, expected_result, keyset_cache):
    artifact_dir = "./py_test_lib_compile_and_run"
    engine = LibrarySupport.new(artifact_dir)
    compile_run_assert(engine, mlir_input, args, expected_result, keyset_cache)
    shutil.rmtree(artifact_dir)


@pytest.mark.parametrize("mlir_input, args, expected_result", end_to_end_fixture)
def test_lib_compile_reload_and_run(mlir_input, args, expected_result, keyset_cache):
    artifact_dir = "./test_lib_compile_reload_and_run"
    engine = LibrarySupport.new(artifact_dir)
    # Here don't save compilation result, reload
    engine.compile(mlir_input)
    compilation_result = engine.reload()
    result = run(engine, args, compilation_result, keyset_cache)
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
    engine = LibrarySupport.new(artifact_dir)
    engine.compile(mlir_str)
    assert os.path.exists(engine.get_program_info_path())
    assert os.path.exists(engine.get_shared_lib_path())
    shutil.rmtree(artifact_dir)
    assert not os.path.exists(engine.get_program_info_path())
    assert not os.path.exists(engine.get_shared_lib_path())


def _test_lib_compile_and_run_with_options(keyset_cache, options):
    mlir_input = """
        func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
            %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]> : tensor<128xi64>
            %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<7>, tensor<128xi64>) -> (!FHE.eint<7>)
            return %1: !FHE.eint<7>
        }
    """
    args = (73,)
    expected_result = 73
    engine = LibrarySupport.new("./py_test_lib_compile_and_run_custom_perror")

    compile_run_assert(engine, mlir_input, args, expected_result, keyset_cache, options)


def test_lib_compile_and_run_p_error(keyset_cache):
    options = CompilationOptions.new("main")
    options.set_p_error(0.00001)
    options.set_display_optimizer_choice(True)
    _test_lib_compile_and_run_with_options(keyset_cache, options)


def test_lib_compile_and_run_global_p_error(keyset_cache):
    options = CompilationOptions.new("main")
    options.set_global_p_error(0.00001)
    options.set_display_optimizer_choice(True)
    _test_lib_compile_and_run_with_options(keyset_cache, options)


def test_lib_compile_and_run_security_level(keyset_cache):
    options = CompilationOptions.new("main")
    options.set_security_level(80)
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
    engine = LibrarySupport.new(artifact_dir)
    options = CompilationOptions.new("main")
    options.set_auto_parallelize(True)
    compile_run_assert(
        engine, mlir_input, args, expected_result, keyset_cache, options=options
    )


# This test was running in JIT mode at first. Problem is now, it does not work with the library
# support. It is not clear to me why, but the dataflow runtime seems to have stuffs dedicated to
# the dropped JIT support... I am cancelling it until further explored.
#
# # FIXME #51
# @pytest.mark.xfail(
#     platform.system() == "Darwin",
#     reason="MacOS have issues with translating Cpp exceptions",
# )
# @pytest.mark.parametrize(
#     "mlir_input, args, expected_result", end_to_end_parallel_fixture
# )
# def test_compile_dataflow_and_fail_run(
#     mlir_input, args, expected_result, keyset_cache, no_parallel
# ):
#     if no_parallel:
#         artifact_dir = "./py_test_compile_dataflow_and_fail_run"
#         engine = LibrarySupport.new(artifact_dir)
#         options = CompilationOptions.new("main")
#         options.set_auto_parallelize(True)
#         with pytest.raises(
#             RuntimeError,
#             match="call: current runtime doesn't support dataflow execution",
#         ):
#             compile_run_assert(
#                 engine, mlir_input, args, expected_result, keyset_cache, options=options
#             )


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
            (np.array([[1, 2, 3, 4], [4, 2, 1, 0], [2, 3, 1, 5]], dtype=np.uint8),),
            np.array([[26, 18], [15, 16], [21, 26]]),
            id="matmul_eint_int_uint8",
        ),
    ],
)
def test_compile_and_run_loop_parallelize(
    mlir_input, args, expected_result, keyset_cache
):
    artifact_dir = "./py_test_compile_and_run_loop_parallelize"
    engine = LibrarySupport.new(artifact_dir)
    options = CompilationOptions.new("main")
    options.set_loop_parallelize(True)
    compile_run_assert(
        engine, mlir_input, args, expected_result, keyset_cache, options=options
    )


@pytest.mark.parametrize(
    "mlir_input, args",
    [
        pytest.param(
            """
            func.func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
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
    engine = LibrarySupport.new(artifact_dir)
    with pytest.raises(
        RuntimeError, match=r"function has arity 2 but is applied to too many arguments"
    ):
        compile_run_assert(engine, mlir_input, args, None, keyset_cache)


@pytest.mark.parametrize(
    "mlir_input",
    [
        pytest.param(
            """
            func.func @test(%arg0: tensor<4x!FHE.eint<7>>, %arg1: tensor<4xi8>) -> !FHE.eint<7>
            {
                %ret = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
                    (tensor<4x!FHE.eint<7>>, tensor<4xi8>) -> !FHE.eint<7>
                return %ret : !FHE.eint<7>
            }
            """,
            id="not @main",
        ),
    ],
)
def test_compile_invalid(mlir_input):
    artifact_dir = "./py_test_compile_invalid"
    engine = LibrarySupport.new(artifact_dir)
    with pytest.raises(RuntimeError, match=r"Function not found, name='main'"):
        engine.compile(mlir_input)


def test_crt_decomposition_feedback():
    mlir = """
    
func.func @main(%arg0: !FHE.eint<16>) -> !FHE.eint<16> {
    %tlu = arith.constant dense<60000> : tensor<65536xi64>
    %1 = "FHE.apply_lookup_table"(%arg0, %tlu): (!FHE.eint<16>, tensor<65536xi64>) -> (!FHE.eint<16>)
    return %1: !FHE.eint<16>
}

    """

    artifact_dir = "./py_test_crt_decomposition_feedback"
    engine = LibrarySupport.new(artifact_dir)
    compilation_result = engine.compile(mlir, options=CompilationOptions.new("main"))
    compilation_feedback = engine.load_compilation_feedback(compilation_result)

    assert isinstance(compilation_feedback, CompilationFeedback)
    assert compilation_feedback.crt_decompositions_of_outputs == [[7, 8, 9, 11, 13]]


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
            {'loc("some/random/location.py":10:2)': 1187400},
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
                'loc("@lut some/random/location.py":11:2)': 597424,
                # 4*2*4097*8 (result buffer)
                'loc("@return some/random/location.py":12:2)': 262208,
            },
            id="multiple location",
        ),
    ],
)
def test_memory_usage(mlir: str, expected_memory_usage_per_loc: dict):
    artifact_dir = "./test_memory_usage"
    engine = LibrarySupport.new(artifact_dir)
    compilation_result = engine.compile(mlir)
    compilation_feedback = engine.load_compilation_feedback(compilation_result)
    assert isinstance(compilation_feedback, CompilationFeedback)

    assert (
        expected_memory_usage_per_loc == compilation_feedback.memory_usage_per_location
    )

    shutil.rmtree(artifact_dir)
