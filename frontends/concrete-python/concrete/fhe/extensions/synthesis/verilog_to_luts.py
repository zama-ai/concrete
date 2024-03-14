"""
Rewrite yosys json output as a simple Tlu Dag.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VERBOSE = False


def log(*args):
    """Log function statically activated."""
    if VERBOSE:
        print(*args)


## Assuming same arity everywhere, i.e. dag with uniform precision TLUs
LUT_COSTS = {
    1: 29,
    2: 33,
    3: 45,
    4: 74,
    5: 101,
    6: 231,
    7: 535,
    8: 1721,
}


def luts_spec_abc():
    """Generate the costs table (area, delay) for `abc`."""
    return "\n".join(
        # arity area delay
        f"{arity}\t{cost}\t{cost}"
        for arity, cost in LUT_COSTS.items()
    )


YOSYS_EXE_NAME = "yowasp-yosys"
_yosys_exe = None  # pylint: disable=invalid-name


def detect_yosys_exe():
    """Detect yosys executable."""
    global _yosys_exe  # noqa: PLW0603 pylint: disable=global-statement
    if _yosys_exe:
        return _yosys_exe
    result = (
        shutil.which(YOSYS_EXE_NAME)
        or shutil.which(YOSYS_EXE_NAME, path=os.path.dirname(sys.executable))
        or shutil.which(YOSYS_EXE_NAME, path=os.path.dirname(shutil.which("python3") or ""))
    )
    if result is None:
        msg = f"{YOSYS_EXE_NAME} cannot be found."
        raise RuntimeError(msg)
    _yosys_exe = result
    return result


def yosys_script(abc_path, verilog_path, json_path, dot_file, no_clean_up=False):
    """Generate `yosys` scripts."""
    no_cleanup = "-nocleanup -showtmp" if no_clean_up else ""
    return f"""
echo on
read -sv {verilog_path};
prep
techmap
log Synthesis with ABC: {abc_path}
abc {no_cleanup} -script {abc_path}
write_json {json_path}
""" + (
        "" if not dot_file else "show -stretch"
    )


def abc_script(lut_cost_path):
    """Generate `abc` scripts."""
    return f"""
# & avoid a bug when cleaning tmp
read_lut {lut_cost_path}
print_lut
strash
&get -n
&fraig -x
&put
scorr
dc2
dretime
strash
dch -f
if
mfs2
lutpack
"""


def bstr(bytes_str):
    """Binary str to str."""
    return str(bytes_str, encoding="utf-8")


def _yosys_run_script(
    abc_file, lut_costs_file, yosys_file, verilog_file, verilog_content, json_file, dot_file=True
):
    """Run the yosys script using a subprocess based on the inputs/outpus files."""
    tmpdir_prefix = Path.home() / ".cache" / "concrete-python" / "synthesis"
    os.makedirs(tmpdir_prefix, exist_ok=True)
    new_files = [
        (abc_file, abc_script(lut_costs_file.name)),
        (lut_costs_file, luts_spec_abc()),
        (verilog_file, verilog_content),
        (yosys_file, yosys_script(abc_file.name, verilog_file.name, json_file.name, dot_file)),
    ]
    for new_file, content in new_files:
        new_file.write(content)
        new_file.flush()
    yosys_call = [detect_yosys_exe(), "-s", yosys_file.name]
    try:
        completed = subprocess.run(yosys_call, check=True, capture_output=True)
        log(completed.stdout)
        log(completed.stderr)
    except subprocess.CalledProcessError as exc:
        log(exc.output)
        log(exc.stderr)
        raise_verilog_warnings_and_error(
            exc.stdout + exc.stderr, verilog_file.name, verilog_content
        )
        print(bstr(exc.output))
        print(bstr(exc.stderr))
        raise

    if b"Warning" in completed.stdout:
        raise_verilog_warnings_and_error(completed.stdout, verilog_file.name, verilog_content)
    try:
        return json.load(json_file)
    except json.decoder.JSONDecodeError:
        if not json_file.read():
            print(completed.stdout)
            print(completed.stderr)
        else:
            print("JSON:", json_file.read())
        raise


def raise_verilog_warnings_and_error(output, verilog_path, verilog_content):
    """Raise a tailored exception to provide user information to the detected error."""
    if isinstance(output, bytes):
        output = str(output, encoding="utf8")
    fatal = None
    msgs = []
    for line in output.splitlines():
        location = verilog_path + ":"
        if line.startswith(location):
            msgs.append(line)
            line_nb = int(line.split(location)[1].split(":")[0])
            lines = verilog_content.splitlines()
            context_lines = "\n".join(verilog_content.splitlines()[line_nb - 2 : line_nb])
            underline = "^" * len(lines[line_nb - 1])
            fatal = f"{line}\nError at line {line_nb}:\n{context_lines}\n{underline}"
        elif "Warning" in line:
            msgs.append(line)
    if msgs and (len(msgs) != 1 and fatal):
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("The following warnings/errors need to be checked")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
        print("\n".join(msgs))
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
    if fatal:
        raise ValueError(fatal) from None


def yosys_run_script(verilog, yosys_dot_file):
    """Run the yosys script with the adequate generated files."""
    tmpdir_prefix = Path.home() / ".cache" / "concrete-python" / "synthesis"
    os.makedirs(tmpdir_prefix, exist_ok=True)

    def tmpfile(mode, suffix, **kwargs):
        return tempfile.NamedTemporaryFile(
            mode=mode, dir=tmpdir_prefix, suffix=f"-{suffix}", **kwargs
        )

    # fmt: off
    with \
            tmpfile("w+", "script.abc") as abc_file, \
            tmpfile("w+", "lut-costs.txt") as lut_costs_file, \
            tmpfile("w+", "source.verilog") as verilog_file, \
            tmpfile("w+", "yosys.script") as yosys_file, \
            tmpfile("r", "luts.json") as json_file:
        return _yosys_run_script(
            abc_file, lut_costs_file, yosys_file,
            verilog_file, verilog,
            json_file, dot_file=yosys_dot_file,
        )
    # fmt: on


@dataclass(frozen=True)
class ValueOrigin:
    """Original verilog input/output information for a value."""

    is_parameter: bool = False
    is_result: bool = False
    bit_index: int = 0
    base_name: str = ""


@dataclass(frozen=True, order=True)
class ValueNode:
    """An intermediate named value."""

    name: str
    origin: ValueOrigin = ValueOrigin()

    @classmethod
    def parameter(cls, base_name, i):
        """Construct a parameter (i.e. verilog input) derived value from the base input name."""
        # E.g. `ValueNode.parameter("a", 3)` is "a[3]", the 4th bit of verilog input `a`.
        return ValueNode(
            f"{base_name}[{i}]", ValueOrigin(is_parameter=True, bit_index=i, base_name=base_name)
        )

    @classmethod
    def result(cls, base_name, i):
        """Construct a result (i.e. verilog input) derived value from the base output name."""
        # E.g. `ValueNode.result("r", 3)` is "r[3]", the 4th bit of verilog output `r`.
        return ValueNode(
            f"{base_name}[{i}]", ValueOrigin(is_result=True, bit_index=i, base_name=base_name)
        )

    @classmethod
    def port(cls, base_name, port, i):
        """Construct a verilog port, i.e. either parameter/input or result/output."""
        if port["direction"] == "input":
            return cls.parameter(base_name, i)
        assert port["direction"] == "output"
        return cls.result(base_name, i)

    @property
    def is_interface(self):
        """Check if a value is part of the verilog circuit interface."""
        return self.origin.is_parameter or self.origin.is_result


@dataclass
class TluNode:
    """A TLU operation node."""

    arguments: 'list[ValueNode]'
    results: 'list[ValueNode]'
    content: 'list[Any]'

    @property
    def arity(self):
        """Number of single bit parameters of the TLU."""
        return len(self.arguments)

    @property
    def name(self):
        """Name of the result."""
        assert len(self.results) == 1
        return self.results[0].name

    def __str__(self):
        name = ", ".join(r.name for r in self.results)
        args = ", ".join(a.name for a in self.arguments)
        return f"{name} = tlu ({args}) ({self.content})"


@dataclass
class TluCircuit:
    """A full circuit composent of parameters, results and intermediates nodes."""

    name: str
    parameters: 'dict[str, list[ValueNode]]'
    results: 'list[list[ValueNode]]'
    nodes: 'list[TluNode]'


def convert_yosys_json_to_circuit(json_data, circuit_name="noname"):
    """Create a Circuit object from yosys json output."""
    modules = json_data["modules"]
    assert len(modules) == 1
    (module,) = modules.values()
    assert set(module.keys()) == {"attributes", "ports", "cells", "netnames"}, module.keys()
    symbolic_name = {0: "0", 1: "1"}
    nodes = []

    symbol_set = set()
    parameters = {}
    results = []
    for name, port in module["ports"].items():
        elements = []
        for i, bit in enumerate(port["bits"]):
            bit_node = ValueNode.port(name, port, i)
            elements.append(bit_node)
            symbol_set.add(bit_node.name)
            if bit in ("0", "1"):
                # output wired to constant
                nodes += [
                    TluNode(
                        arguments=[],
                        results=[bit_node],
                        content=[bit],
                    )
                ]
            elif bit in symbolic_name:
                # input wired to output
                nodes += [
                    TluNode(
                        arguments=[symbolic_name[bit]],
                        results=[bit_node],
                        content=[0, 1],
                    )
                ]
                log("Equiv: ", symbolic_name[bit], bit_node)
            else:
                symbolic_name[bit] = bit_node
        if elements[0].origin.is_parameter:
            parameters[name] = elements
        else:
            results += [elements]

    log("Interface Names:", symbolic_name)

    for cell_value in module["cells"].values():
        assert cell_value["type"] == "$lut", cell_value
        content = list(reversed(list(map(int, cell_value["parameters"]["LUT"]))))
        assert set(cell_value["connections"].keys()) == {"A", "Y"}
        arguments = [
            symbolic_name.get(n, ValueNode(f"n{n}")) for n in cell_value["connections"]["A"]
        ]
        # the first input is the last index # missing transpose ?
        structured_content = np.array(content).reshape((2,) * (len(arguments))).tolist()

        cell_results = [
            symbolic_name.get(n, ValueNode(f"n{n}")) for n in cell_value["connections"]["Y"]
        ]
        nodes += [TluNode(arguments=arguments, results=cell_results, content=structured_content)]

    for node in nodes:
        log(len(node.arguments), node.arguments, "---", node.content, "-->", node.results)

    log("Nodes:", nodes)
    return TluCircuit(circuit_name, parameters, results, nodes)


def yosys_lut_synthesis(verilog: str, yosys_dot_file=False, circuit_name="noname"):
    """Create a Circuit object from a verilog module."""
    json_data = yosys_run_script(verilog, yosys_dot_file)
    return convert_yosys_json_to_circuit(json_data, circuit_name)
