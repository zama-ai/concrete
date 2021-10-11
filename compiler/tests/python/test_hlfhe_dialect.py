import pytest
from mlir.ir import Context
from zamalang import register_dialects
from zamalang.dialects import hlfhe


def test_eint():
    ctx = Context()
    register_dialects(ctx)
    eint = hlfhe.EncryptedIntegerType.get(ctx, 6)
    assert eint.__str__() == "!HLFHE.eint<6>"


# FIXME: need to handle error on call to hlfhe.EncryptedIntegerType.get and throw an exception to python
# def test_invalid_eint():
#     ctx = Context()
#     register_dialects(ctx)
#     with pytest.raises(RuntimeError, match=r"mlir parsing failed"):
#         eint = hlfhe.EncryptedIntegerType.get(ctx, 16)
