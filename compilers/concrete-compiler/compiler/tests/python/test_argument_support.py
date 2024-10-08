import pytest
import numpy as np
from concrete.compiler.utils import ACCEPTED_NUMPY_UINTS
from concrete.compiler import Value


@pytest.mark.parametrize(
    "garbage",
    [
        pytest.param(None, id="None"),
        pytest.param([0, 1, 2], id="list"),
        pytest.param(0.5, id="float"),
        pytest.param(2**70, id="large int"),
        pytest.param("aze", id="str"),
        pytest.param(np.float64(0.8), id="np.float64"),
    ],
)
def test_invalid_arg_type(garbage):
    with pytest.raises(RuntimeError):
        Value(garbage)


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
        arg = Value(value)
    except Exception:
        pytest.fail(f"value of type {type(value)} should be supported")
    assert arg.is_scalar(), "should have been a scalar"
    assert arg.to_py_val() == value


# TODO: #495
@pytest.mark.parametrize(
    "dtype, maxvalue",
    [
        pytest.param(np.uint8, 2**8 - 1, id="uint8"),
        pytest.param(np.uint16, 2**16 - 1, id="uint16"),
        pytest.param(np.uint32, 2**32 - 1, id="uint32"),
        pytest.param(np.uint64, 2**64 - 1, id="uint64"),
    ],
)
def test_accepted_ndarray(dtype, maxvalue):
    value = np.array([0, 1, 2, maxvalue], dtype=dtype)
    try:
        arg = Value(value)
    except Exception:
        pytest.fail(f"value of type {type(value)} should be supported")

    assert arg.is_tensor(), "should have been a tensor"
    assert np.all(np.equal(arg.get_shape(), value.shape))
    assert np.all(
        np.equal(
            value,
            np.array(arg.to_py_val()),
        )
    )


def test_accepted_array_as_scalar():
    value = np.array(7, dtype=np.uint16)
    try:
        arg = Value(value)
    except Exception:
        pytest.fail(f"value of type {type(value)} should be supported")
    assert arg.is_scalar(), "should have been a scalar"
    assert arg.to_py_val() == value
