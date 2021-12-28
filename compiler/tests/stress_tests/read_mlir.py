from dataclasses import dataclass
import re

from stress_tests.utils import CONCRETECOMPILER, log2, ceil_log2, run


DUMP_HLFHE = '--action=dump-hlfhe'
DUMP_LOWLFHE =  '--action=dump-lowlfhe'

def read_max_mlir_attribute(name, content):
    regexp = re.compile(f'{name} = (?P<value>[0-9]+)')
    return max(
        int(found.group('value'))
        for found in regexp.finditer(content)
    )

def log_manp_max(path):
    hlfhe = run(CONCRETECOMPILER, path, DUMP_HLFHE)
    return ceil_log2(read_max_mlir_attribute('MANP', hlfhe))

@dataclass
class FHEParams:
    log_poly_size: int
    glwe_dim: int

def v0_param(path):
    lowlfhe = run(CONCRETECOMPILER, path, DUMP_LOWLFHE)
    return FHEParams(
        log_poly_size=log2(read_max_mlir_attribute('polynomialSize', lowlfhe)),
        glwe_dim=read_max_mlir_attribute('glweDimension', lowlfhe),
    )
