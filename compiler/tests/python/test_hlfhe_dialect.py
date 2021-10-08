from mlir.ir import Context
from zamalang import register_dialects
from zamalang.dialects import hlfhe


def test_eint():
    ctx = Context()
    register_dialects(ctx)
    eint = hlfhe.EncryptedIntegerType.get(ctx, 6)
    assert eint.__str__() == "!HLFHE.eint<6>"
