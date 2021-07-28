import zamalang
import zamalang.dialects.hlfhe as hlfhe
import mlir.dialects.builtin as builtin
import mlir.dialects.std as std
from mlir.ir import *


def main():
    with Context() as ctx, Location.unknown():
        # register zamalang's dialects
        zamalang.register_dialects(ctx)

        module = Module.create()
        eint16 = hlfhe.EncryptedIntegerType.get(ctx, 16)
        with InsertionPoint(module.body):
            func_types = [RankedTensorType.get((10, 10), eint16) for _ in range(2)]
            @builtin.FuncOp.from_py_func(*func_types)
            def fhe_circuit(*arg):
                return arg[0]

        print(module)


if __name__ == "__main__":
    main()