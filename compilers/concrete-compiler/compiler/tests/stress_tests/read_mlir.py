from dataclasses import dataclass
import re

from stress_tests.utils import CONCRETECOMPILER, log2, ceil_log2, run


DUMP_FHE = '--action=dump-fhe'
DUMP_CONCRETE =  '--action=dump-concrete'

def read_max_mlir_attribute(name, content):
    regexp = re.compile(f'{name} = (?P<value>[0-9]+)')
    return max(
        int(found.group('value'))
        for found in regexp.finditer(content)
    )

def log_manp_max(path):
    fhe = run(CONCRETECOMPILER, path, DUMP_FHE)
    return ceil_log2(read_max_mlir_attribute('MANP', fhe))

@dataclass
class FHEParams:
    log_poly_size: int
    glwe_dim: int

def v0_param(path):
    concrete = run(CONCRETECOMPILER, path, DUMP_CONCRETE)
    return FHEParams(
        log_poly_size=log2(read_max_mlir_attribute('polynomialSize', concrete)),
        glwe_dim=read_max_mlir_attribute('glweDimension', concrete),
    )
