import os
import tempfile

import pytest
import numpy as np
from concrete.compiler import CompilerEngine

KEY_SET_CACHE_PATH = os.path.join(tempfile.gettempdir(), "KeySetCache")


@pytest.mark.parallel
@pytest.mark.parametrize(
    "mlir_input, args, expected_result",
    [
        pytest.param(
            """
            func @main(%arg0: !FHE.eint<7>, %arg1: i8) -> !FHE.eint<7> {
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
            func @main(%arg0: tensor<4x!FHE.eint<7>>, %arg1: tensor<4xi8>) -> !FHE.eint<7>
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
    ],
)
def test_compile_and_run_parallel(mlir_input, args, expected_result):
    engine = CompilerEngine()
    engine.compile_fhe(
        mlir_input,
        unsecure_key_set_cache_path=KEY_SET_CACHE_PATH,
        auto_parallelize=True,
    )
    if isinstance(expected_result, int):
        assert engine.run(*args) == expected_result
    else:
        # numpy array
        assert np.all(engine.run(*args) == expected_result)
