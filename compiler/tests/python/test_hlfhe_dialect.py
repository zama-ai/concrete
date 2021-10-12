import pytest
from mlir.ir import Context
from zamalang import register_dialects
from zamalang.dialects import hlfhe


@pytest.mark.parametrize("width", list(range(1, 8)))
def test_eint(width):
    ctx = Context()
    register_dialects(ctx)
    eint = hlfhe.EncryptedIntegerType.get(ctx, width)
    assert eint.__str__() == f"!HLFHE.eint<{width}>"


@pytest.mark.parametrize("width", [0, 8, 10, 12])
def test_invalid_eint(width):
    ctx = Context()
    register_dialects(ctx)
    with pytest.raises(ValueError, match=r"can't create eint with the given width"):
        eint = hlfhe.EncryptedIntegerType.get(ctx, width)
