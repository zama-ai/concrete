import contextlib
import concurrent.futures as futures
from itertools import chain
import json
import os
from tempfile import gettempdir

import pytest

from stress_tests.experiment import (
    ExperimentConditions, Experiment, Encoder, Replication
)
from stress_tests import read_mlir
from stress_tests.utils import CONCRETECOMPILER, run
from stress_tests.v0_parameters import P_MAX, LOG2_MANP_MAX

POSSIBLE_BITWIDTH = range(1, P_MAX+1)
POSSIBLE_SIZE = range(1, 128)

TEST_PATH = os.path.dirname(__file__)
TRACE = os.path.join(TEST_PATH, 'trace')

JIT_INVOKE_MAIN = (
    '--action=jit-invoke',
    '--funcname=main',
    '--jit-keyset-cache-path=/tmp/StresstestsCache',
)

def jit_args(*params):
    return tuple(
        f'--jit-args={p}' for p in params
    )

CONTROLLED_CODE_PARAMS = sorted(chain.from_iterable(
    {   #(bitwidth, size, input value)
        (bitwidth, POSSIBLE_SIZE[-1], 0),
        (bitwidth, 1, 1),
        (bitwidth, bitwidth, 1),
        (bitwidth, 2 ** (bitwidth - 2), 1),
        (bitwidth, 2 ** (bitwidth - 1), 1),
        (bitwidth, 2 ** bitwidth - 1, 1),
        (bitwidth, 2 ** bitwidth, 1), # force carry
        (bitwidth, 2 ** (bitwidth+1), 1), # force overflow and carry 0 ?
    }# <-- a set to deduplicate similar cases
    for bitwidth in POSSIBLE_BITWIDTH
))


CONTROLLED_CODE_PARAMS = [
    case for case in CONTROLLED_CODE_PARAMS
    if case[1] >= 1
]
TEST_CONTROLLED_REPLICATE = 100

WILD_CODE_PARAMS = list(sorted(chain.from_iterable(
    {   #(bitwidth, size, input value)
        (bitwidth, 2 ** bitwidth + 8, 1),
        (bitwidth, 2 ** bitwidth + 9, 1),
        (bitwidth, 2 ** bitwidth + 16, 1),
        (bitwidth, 2 ** bitwidth + 17, 1),
        (bitwidth, 2 ** (2 * bitwidth), 1),
        (bitwidth, 2 ** (2 * bitwidth) + 1, 1),
    }# <-- a set to deduplicate similar cases
    for bitwidth in POSSIBLE_BITWIDTH
)))
TEST_WILD_RETRY = 3

def basic_multisum_identity(bitwidth, size):
    def components(name, size, ty=''):
        ty_annot = ' : ' + ty if ty else ''
        return ', '.join(f'%{name}{i}{ty_annot}' for i in range(size))
    def tensor(size, ty):
        return f'tensor<{size}x{ty}>'
    v_ty = f"!FHE.eint<{bitwidth}>"
    tv_ty = tensor(size, v_ty)
    w_ty = f"i{bitwidth+1}"
    w_modulo = 2 ** bitwidth # to match v bitwidth
    tw_ty = tensor(size, w_ty)
    lut_size = 2**bitwidth
    lut_ty = 'i64'
    tlut_ty = tensor(lut_size, lut_ty)

    return (
f"""
func.func @main({components('v', size, v_ty)}) -> {v_ty} {{
  %v = tensor.from_elements {components('v', size)} : {tv_ty}

  // Declare {size} %wX components
  { ''.join(f'''
  %w{i} = arith.constant 1: {w_ty}'''
    for i in range(size)
  )}
  %w = tensor.from_elements {components('w', size)} : {tw_ty}

  // Declare {lut_size}  %lutX components
  { ''.join(f'''
  %lut{i} = arith.constant {i}: i64'''
    for i in range(lut_size)
  )}
  %lut = tensor.from_elements {components('lut', lut_size)} : {tlut_ty}

  %dot_product = "FHELinalg.dot_eint_int"(%v, %w) : ({tv_ty}, {tw_ty}) -> {v_ty}
  %pbs_result = "FHE.apply_lookup_table"(%dot_product, %lut): ({v_ty}, {tlut_ty}) -> {v_ty}
  return %pbs_result: {v_ty}
}}
"""
    )


executor = futures.ThreadPoolExecutor()

def basic_setup(bitwidth, size, const, retry=10):
    code = basic_multisum_identity(bitwidth, size)
    args = (const,) * size
    expected = eval_basic_multisum_identity(bitwidth, args)
    with tmp_file(f'basic_{bitwidth:03}_{size:03}_{const}.mlir', code) as path:
        modulo = 2 ** bitwidth
        # Read various value from compiler
        log_manp_max = read_mlir.log_manp_max(path)
        params = read_mlir.v0_param(path)

        conditions_details = []
        def msg(m, append_here=None, space=' '):
            print(m, end=space, flush=True)  # test human output
            if append_here is not None:
                append_here.append(m)

        if (LOG2_MANP_MAX < log_manp_max):
            msg('HIGH-MANP', conditions_details)
        if 2 ** bitwidth <= expected:
            msg(f'OVERFLOW', conditions_details)

        cmd = (CONCRETECOMPILER, path) + JIT_INVOKE_MAIN + jit_args(*args)
        compilers_calls = [executor.submit(run, *cmd) for _ in range(retry)]

        success = 0
        overflow = 0
        replications = []
        for replication in futures.as_completed(compilers_calls):
            result = int(replication.result().splitlines()[-1])
            correct_in_modulo = expected % modulo == result % modulo
            details = []
            replications.append(Replication(correct_in_modulo, details))
            if not (0 <= result < modulo):
                msg(f'OVERFLOW {result}', details)
                overflow += 1
            if correct_in_modulo:
                msg('O', space='')
                success += 1
            else:
                msg('X', space='')
                diff = f'Expected :{expected % modulo} vs. {result % modulo} (no modulo {expected} vs. {result}'
                details.append(diff)

        print(' ', end='')
        add_to(TRACE, Experiment(
            cmd = ' '.join(cmd),
            conditions=ExperimentConditions(
                bitwidth=bitwidth, size=size, args=args,
                log_manp_max=log_manp_max,
                overflow=2 ** bitwidth <= expected,
                details=conditions_details,),
            replications=replications,
            code=code,
            success_rate=100.0 * success/retry,
            overflow_rate=100.0 * overflow/retry,
        ))

        assert success == len(replications)


def eval_basic_multisum_identity(bitwidth, args):
    return sum(
        arg
        for arg in args
    )

@contextlib.contextmanager
def tmp_file(name, content, delete=False):
    path = os.path.join(gettempdir(), 'stresstests', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    yield f.name
    if delete:
        os.remove()

def add_to(DIR, expe: Experiment):
    full_test_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    test_name = full_test_name.rsplit('[', 1)[0]
    DIR = os.path.join(DIR, test_name)
    os.makedirs(DIR, exist_ok=True)
    conditions = expe.conditions
    name = f'{conditions.bitwidth:03}bits_x_{conditions.size:03}_{conditions.args[0]}'
    with open(os.path.join(DIR, name), 'w') as f:
         json.dump(expe, f, indent=2, cls=Encoder)


@pytest.mark.parametrize("bitwidth, size, const", CONTROLLED_CODE_PARAMS)
def test_controlled(bitwidth, size, const):
    return basic_setup(bitwidth, size, const, TEST_CONTROLLED_REPLICATE)

@pytest.mark.parametrize("bitwidth, size, const", WILD_CODE_PARAMS)
def test_wild(bitwidth, size, const):
    return basic_setup(bitwidth, size, const, TEST_WILD_RETRY)
