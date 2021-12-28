import pytest
from mlir.ir import Context, RankedTensorType, Location
from concrete.lang import register_dialects
from concrete.lang.dialects import hlfhe


@pytest.mark.parametrize("width", list(range(1, 8)))
def test_eint(width):
    ctx = Context()
    register_dialects(ctx)
    eint = hlfhe.EncryptedIntegerType.get(ctx, width)
    assert eint.__str__() == f"!HLFHE.eint<{width}>"


@pytest.mark.parametrize("shape", [(1,), (2,), (1, 1), (1, 2), (2, 1), (3, 3, 3)])
def test_eint_tensor(shape):
    with Context() as ctx, Location.unknown(context=ctx):
        register_dialects(ctx)
        eint = hlfhe.EncryptedIntegerType.get(ctx, 3)
        tensor = RankedTensorType.get(shape, eint)
        assert (
            tensor.__str__() == f"tensor<{'x'.join(map(str, shape))}x!HLFHE.eint<{3}>>"
        )


@pytest.mark.parametrize("width", [0, 8, 10, 12])
def test_invalid_eint(width):
    ctx = Context()
    register_dialects(ctx)
    with pytest.raises(ValueError, match=r"can't create eint with the given width"):
        eint = hlfhe.EncryptedIntegerType.get(ctx, width)
