# Interactive Debugger

The Concrete Interactive Debugger lets you inspect what happens inside your FHE circuits. Instead of treating a circuit as a black box, you can see every intermediate value, detect overflows, compare cleartext against simulation, and step through execution in VS Code.

## Inspecting intermediate values

`circuit.inspect()` evaluates your circuit in cleartext (no encryption, no noise) and returns a snapshot of every node's computed value.

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def add(x, y):
    return x + y

circuit = add.compile([(i, j) for i in range(8) for j in range(8)])

result = circuit.inspect(2, 6)
print(result.summary())
```

Output:

```
%0 = x            => 2
%1 = y            => 6
%2 = add(%0, %1)  => 8
return %2
```

Access the final output directly with `result.output`.

### Overflow detection

When values exceed the range the circuit was compiled for, `inspect()` tells you exactly which nodes overflowed:

```python
bad = circuit.inspect(100, 100)
print(bad.has_overflow)  # True

for snap in bad.overflows:
    print(snap)
```

### Stopping early

Halt evaluation before a specific node — like a breakpoint:

```python
# Stop by predicate
result = circuit.inspect(2, 6, stop_at=lambda node: node.properties.get("name") == "add")

# Stop by source location
result = circuit.inspect(2, 6, stop_at="/path/to/my_file.py:42")
```

When stopped early, `result.output` raises `RuntimeError` since the final output was never computed.

### Filtering snapshots

```python
result = circuit.inspect(2, 6)

result.filter(operation_filter="add")
result.filter(tag_filter="my_tag")
result.filter(is_encrypted_filter=True)
result.filter(overflow_only=True)
result.filter(custom_filter=lambda snap: snap.value > 5)
```

### Snapshot properties

Each `NodeSnapshot` carries:

| Property | Description |
|----------|-------------|
| `snap.value` | Computed value at this node |
| `snap.index` | Position in evaluation order |
| `snap.operation_name` | `"input"`, `"add"`, `"multiply"`, etc. |
| `snap.location` | Source file and line number |
| `snap.tag` | User-assigned tag (from `fhe.tag(...)`) |
| `snap.is_encrypted` | Whether the output is encrypted |
| `snap.overflow` | Whether the value exceeds the dtype range |

For multi-function modules, inspect per-function:

```python
module.my_func.inspect(x)
```

---

## Simulation probes

`inspect()` evaluates in pure Python. To verify what happens in the actual compiled MLIR simulation pipeline, use `circuit.run_with_probes()`. This injects debug probe ops into the MLIR, runs simulation, and captures values at each probed node.

```python
@fhe.compiler({"x": "encrypted"})
def f(x):
    return (x + 1) * 2

circuit = f.compile(range(8))

probed = circuit.run_with_probes(5)
print(probed.output)    # 12
print(probed.summary())
```

Output:

```
Output: 12
Probes: 2

  ID  Operation             Tag                   Value  Overflow
--------------------------------------------------------------------------------
   1  add                                             6
   2  multiply                                       12
```

### Comparing cleartext vs. simulation

```python
inspection = circuit.inspect(5)
probed = circuit.run_with_probes(5)

print(probed.compare_with(inspection))
```

```
Node                            Inspect (cleartext)    Probe (simulation)  Match
----------------------------------------------------------------------------------------------------
input                                             5                     -  (no probe)
add                                               6                     6  OK
multiply                                         12                    12  OK
```

If they disagree, the `Match` column shows `MISMATCH`.

### Choosing which nodes to probe

```python
# All encrypted nodes (default)
probed = circuit.run_with_probes(5)

# By tag
probed = circuit.run_with_probes(5, probes=["step1"])

# By predicate
probed = circuit.run_with_probes(5, probes=lambda node: node.converted_to_table_lookup)
```

Filtering and overflow detection work the same way as `inspect()`:

```python
probed.filter(tag_filter="step1")
probed.has_overflow
probed.overflows
```

Each `ProbeSnapshot` has the same properties as `NodeSnapshot`, plus `snap.probe_id`.

For modules: `module.my_func.run_with_probes(x, probes=["my_tag"])`.

> **Note:** Each `run_with_probes()` call recompiles the MLIR. This is fast for small circuits but may be noticeable for large ones.

---

## VS Code debugger

The VS Code extension gives you a standard IDE debugging experience for FHE circuits: breakpoints, stepping, a variables pane, and a debug console.

### Installation

Build and package the extension:

```bash
cd tools/vscode-concrete-debugger
npm install
npm run build
npx vsce package
```

In VS Code: **Cmd+Shift+P** → **"Extensions: Install from VSIX..."** → select the `.vsix` file.

### Launch configuration

Add to your `.vscode/launch.json`:

```jsonc
{
    "type": "concrete",
    "request": "launch",
    "name": "Debug FHE Circuit",
    "program": "${file}",
    "function": "my_circuit",       // variable name of the compiled Circuit
    "args": [3, 5],                 // input values
    "pythonPath": "python3",        // must have concrete-python installed
    "stopOnEntry": true,            // pause before first operation
    "stopOnOverflow": false         // stop on bit-width overflow
}
```

For `@fhe.module`:

```jsonc
{
    "type": "concrete",
    "request": "launch",
    "name": "Debug FHE Module",
    "program": "${file}",
    "function": "my_module",
    "functions": [
        { "name": "scale", "args": [5] },
        { "name": "shift", "args": [7] }
    ],
    "stopOnEntry": true
}
```

### Example script

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def my_circuit(x, y):
    with fhe.tag("compute"):
        return (x + y) * 2

inputset = [(i, j) for i in range(8) for j in range(8)]
my_circuit = my_circuit.compile(inputset)
```

Open the script, select your launch configuration, and press **F5**.

### Stepping

| Action | Key | Behavior |
|---|---|---|
| Step Over | F10 | Evaluate next DAG node (groups nodes from the same source line) |
| Continue | F5 | Run to next breakpoint or end |
| Step Into | F11 | Enter next function (`@fhe.module`) or step one node |
| Step Out | Shift+F11 | Finish current function |

### Breakpoints

Set breakpoints on Python source lines as usual. The debugger maps each line to DAG nodes at that location. A breakpoint shows as a solid red dot if at least one node exists at that line.

### Variables pane

When stopped, two scopes are available:

**Current Node:**

| Variable | Example |
|---|---|
| `value` | `42` |
| `operation` | `add` |
| `encrypted` | `True` |
| `bit_width` | `8` |
| `overflow` | `False` |
| `tag` | `compute` |
| `location` | `script.py:12` |
| `bounds` | `[0, 255]` |

**All Evaluated** — expandable list of every node snapshot so far.

Large arrays show a summary and can be expanded to see individual elements.

### Call stack

Synthesized from the tag hierarchy. A node tagged `layer1.matmul.relu` shows:

```
relu   [layer1.matmul.relu]    @ script.py:12
matmul [layer1.matmul]         @ script.py:12
layer1 [layer1]                @ script.py:12
```

### Debug console

| Expression | Result |
|---|---|
| `value` | Current node's computed value |
| `nodes` | Total nodes and how many evaluated |
| `overflow` | Summary of overflow nodes |
| `snap[3]` | Snapshot at index 3 |
| `function` | Current function name (modules) |
| `functions` | All function names (modules) |

---

## API reference

### `Circuit.inspect(*args, stop_at=None)`

Evaluate in cleartext and return an `InspectionResult`.

- `stop_at` — `str` (location prefix) or `Callable[[Node], bool]`

### `InspectionResult`

| Method / Property | Returns |
|---|---|
| `result.output` | Final output (raises `RuntimeError` if stopped early) |
| `result.has_overflow` | `True` if any node overflowed |
| `result.overflows` | List of overflow snapshots |
| `result.filter(...)` | Query by tag, operation, location, encryption, overflow, or predicate |
| `result.summary()` | Formatted table string |
| `len(result)`, `result[i]`, `for snap in result` | List-like access |

### `Circuit.run_with_probes(*args, probes=None)`

Run in simulation mode with MLIR debug probes and return a `ProbeResult`.

- `probes` — `None` (all encrypted nodes), `list[str]` (by tag), or `Callable[[Node], bool]`

### `ProbeResult`

| Method / Property | Returns |
|---|---|
| `probed.output` | Simulation output |
| `probed.has_overflow` | `True` if any probe overflowed |
| `probed.overflows` | List of overflow snapshots |
| `probed.filter(...)` | Same filtering as `InspectionResult` |
| `probed.summary()` | Formatted table |
| `probed.compare_with(inspection)` | Side-by-side comparison with `InspectionResult` |
| `len(probed)`, `probed[i]`, `for snap in probed` | List-like access |
