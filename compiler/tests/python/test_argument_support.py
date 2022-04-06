import pytest
import numpy as np
from concrete.compiler.utils import ACCEPTED_NUMPY_UINTS
from concrete.compiler import ClientSupport


@pytest.mark.parametrize(
    "garbage",
    [
        pytest.param(None, id="None"),
        pytest.param([0, 1, 2], id="list"),
        pytest.param(0.5, id="float"),
        pytest.param(2**70, id="large int"),
        pytest.param(-8, id="negative int"),
        pytest.param("aze", id="str"),
        pytest.param(np.float64(0.8), id="np.float64"),
        pytest.param(np.int8(9), id="np.int8"),
        pytest.param(np.array([1, 2, 3], dtype=np.int64), id="np.array(np.int64)"),
    ],
)
def test_invalid_arg_type(garbage):
    with pytest.raises(TypeError):
        ClientSupport._create_lambda_argument(garbage)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(5, id="int"),
        pytest.param(np.uint8(5), id="uint8"),
        pytest.param(np.uint16(7), id="uint16"),
        pytest.param(np.uint32(9), id="uint32"),
        pytest.param(np.uint64(1), id="uint64"),
    ],
)
def test_accepted_ints(value):
    try:
        arg = ClientSupport._create_lambda_argument(value)
    except Exception:
        pytest.fail(f"value of type {type(value)} should be supported")
    assert arg.is_scalar(), "should have been a scalar"
    assert arg.get_scalar() == value


# TODO: #495
# @pytest.mark.parametrize(
#     "dtype",
#     [
#         pytest.param(np.uint8, id="uint8"),
#         pytest.param(np.uint16, id="uint16"),
#         pytest.param(np.uint32, id="uint32"),
#         pytest.param(np.uint64, id="uint64"),
#     ],
# )
# def test_accepted_ndarray(dtype):
#     value = np.array([0, 1, 2], dtype=dtype)
#     try:
#         arg = ClientSupport._create_lambda_argument(value)
#     except Exception:
#         pytest.fail(f"value of type {type(value)} should be supported")

#     assert arg.is_tensor(), "should have been a tensor"
#     assert np.all(np.equal(arg.get_tensor_shape(), value.shape))
#     assert np.all(
#         np.equal(
#             value,
#             np.array(arg.get_tensor_data()).reshape(arg.get_tensor_shape()),
#         )
#     )


def test_accepted_array_as_scalar():
    value = np.array(7, dtype=np.uint16)
    try:
        arg = ClientSupport._create_lambda_argument(value)
    except Exception:
        pytest.fail(f"value of type {type(value)} should be supported")
    assert arg.is_scalar(), "should have been a scalar"
    assert arg.get_scalar() == value
