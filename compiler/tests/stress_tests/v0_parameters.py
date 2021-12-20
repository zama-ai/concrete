""" Read parameters matrix in V0Parameters.cpp """
from dataclasses import dataclass
import re

@dataclass
class V0Parameter:
  glweDimension: int
  logPolynomialSize:int
  nSmall: int
  brLevel: int
  brLogBase: int
  ksLevel: int
  ksLogBase:int

# [log_manp][bitwidth]
v0_parameters : 'list[list[V0Parameter]]' = []

def v0_parameter(log_manp_max, bitwidth):
    try:
        return v0_parameters[log_manp_max - 1][bitwidth - 1]
    except IndexError:
        return V0Parameter( *( ['out_of_V0Parameters'] * 7) )

# relative to Makefile
V0Parameters_PATH = 'lib/Support/V0Parameters.cpp'

def read_CPP_decl(name, cpp_filepath):
    DECLARE = re.compile(f'{name}[^=\n]+=')
    END = re.compile(';')
    with open(cpp_filepath) as f:
        content = f.read()
    decl = DECLARE.search(content)
    if not decl:
        raise NameError(f'Cannot find {name} declaration in file {cpp_filepath}')
    end = END.search(content, decl.end())
    assert end
    value = content[decl.end()+1:end.end()-1].strip()
    # print(f'{name} = {value}', decl.group())
    return value

LOG2_MANP_MAX = int(read_CPP_decl('NORM2_MAX', V0Parameters_PATH))
P_MAX = int(read_CPP_decl('P_MAX', V0Parameters_PATH))

try:
    parameters_cpp = read_CPP_decl('parameters', V0Parameters_PATH)
except (FileNotFoundError, NameError) as exc:
    print(exc)
    assert False

v0_parameters = eval(parameters_cpp.replace('{', '[').replace('}', ']'))
