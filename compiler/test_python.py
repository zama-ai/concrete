import concretelang
import concretelang.dialects.fhe as fhe
import mlir.dialects.builtin as builtin
import mlir.dialects.std as std
from mlir.ir import *


def main():
    with Context() as ctx, Location.unknown():
        # register concretelang's dialects
        concretelang.register_dialects(ctx)

        module = Module.create()
        eint6 = fhe.EncryptedIntegerType.get(ctx, 6)
        with InsertionPoint(module.body):
            func_types = [MemRefType.get((10, 10), eint6) for _ in range(2)]
            @builtin.FuncOp.from_py_func(*func_types)
            def main(*arg):
                return arg[0]

        print(module)
        m = """
        func @main(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
            %0 = constant 1 : i3
            %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
            return %1: !FHE.eint<2>
        }"""
        ## Working when HFLFHE and MLIR aren't linked
        concretelang.compiler.round_trip("module{}")
        concretelang.compiler.round_trip(str(module))
        ## END OF WORKING
        ## Doesn't work yet for both modules
        engine = concretelang.CompilerEngine()
        engine.compile_fhe(m)
        # engine.compile_fhe(str(module))
        print(engine.run(2))


if __name__ == "__main__":
    main()