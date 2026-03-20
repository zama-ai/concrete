# Interactive Debugger

The Concrete Interactive Debugger lets you inspect what happens inside your FHE circuits. Instead of treating a circuit as a black box that returns a final answer, you can see every intermediate value, detect overflows, and pause evaluation at any point.

This document covers all debugger features as they are added.

## Phase 1: Simulation Value Inspector

### The problem

Today, debugging an FHE circuit looks like this:

```python
result = circuit.encrypt_run_decrypt(2, 6)  # → 8... but what happened inside?
```

If you get a wrong result, an overflow, or unexpected behavior, there is no way to see what happened at each step. You have to recompile with different configurations and read text dumps.

### The solution: `circuit.inspect()`

`inspect()` evaluates your circuit in cleartext (no encryption, no noise) and returns a snapshot of every node's computed value.

```python
from concrete import fhe

def add(x, y):
    return x + y

compiler = fhe.Compiler(add, {"x": "encrypted", "y": "encrypted"})
inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]
circuit = compiler.compile(inputset)

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

Every node is shown with its computed value. You can also access the final output directly:

```python
result.output  # 8
```

### Overflow detection

When you pass values outside the range the circuit was compiled for, `inspect()` tells you exactly which nodes overflowed and by how much.

```python
# Circuit was compiled for inputs 0-7, so 100 overflows the allocated bit-width
bad = circuit.inspect(100, 100)
print(bad.has_overflow)  # True

for snap in bad.overflows:
    print(snap)
    # NodeSnapshot(index=2, op=add, value=200, overflow=True)

print(bad.summary())
```

The summary marks overflow nodes:

```
%0 = x            => 100    OVERFLOW [100, 100]
%1 = y            => 100    OVERFLOW [100, 100]
%2 = add(%0, %1)  => 200    OVERFLOW [200, 200]
return %2
```

### Stopping early with `stop_at`

For larger circuits, you can halt evaluation before a specific node — like setting a breakpoint.

**Stop by operation name:**

```python
result = circuit.inspect(2, 6, stop_at=lambda node: node.properties.get("name") == "add")
print(result.summary())
```

This shows the state just before the `add` node runs — only inputs are evaluated.

**Stop by source location:**

```python
result = circuit.inspect(2, 6, stop_at="/path/to/my_file.py:42")
```

This halts before any node whose source location starts with that string. Input nodes are never stopped at.

When evaluation is stopped early, accessing `result.output` raises a `RuntimeError` since the final output was never computed.

### Filtering snapshots

You can query the inspection result to find specific nodes:

```python
result = circuit.inspect(2, 6)

# By operation name
result.filter(operation_filter="add")

# By tag
result.filter(tag_filter="my_tag")

# By encryption status
result.filter(is_encrypted_filter=True)

# Overflows only
result.filter(overflow_only=True)

# Custom predicate
result.filter(custom_filter=lambda snap: snap.value > 5)
```

### Browsing individual snapshots

Each `NodeSnapshot` carries:

| Property | Description |
|----------|-------------|
| `snap.value` | The computed value at this node |
| `snap.index` | Position in topological (evaluation) order |
| `snap.operation_name` | `"input"`, `"constant"`, `"add"`, `"multiply"`, etc. |
| `snap.location` | Source file and line number |
| `snap.tag` | User-assigned tag (from `fhe.tag(...)`) |
| `snap.is_encrypted` | Whether the node output is encrypted |
| `snap.overflow` | Whether the value exceeds the node's dtype range |
| `snap.overflow_min` | Actual min of value (if overflow) |
| `snap.overflow_max` | Actual max of value (if overflow) |
| `snap.exceeds_bounds` | Whether the value is outside measured input bounds |

Iterate all snapshots:

```python
for snap in result:
    print(snap.index, snap.operation_name, snap.value)
```

### API reference

**`Circuit.inspect(*args, stop_at=None)`** / **`FheFunction.inspect(*args, stop_at=None)`**

Evaluate the circuit graph in cleartext and return an `InspectionResult`.

- `*args` — input values (same as you'd pass to `encrypt()`)
- `stop_at` — optional `str` (location prefix) or `Callable[[Node], bool]` (predicate); halts before the first matching non-input node

**`InspectionResult`**

| Method / Property | Returns |
|---|---|
| `len(result)` | Number of evaluated nodes |
| `result[i]` | `NodeSnapshot` at index `i` |
| `for snap in result` | Iterate all snapshots |
| `result.output` | Final output value(s); raises `RuntimeError` if stopped early |
| `result.has_overflow` | `True` if any node overflowed |
| `result.overflows` | List of `NodeSnapshot` with overflow |
| `result.filter(...)` | Query snapshots by tag, operation, location, encryption, overflow, or custom predicate |
| `result.summary(...)` | Formatted table string with values and overflow markers |

### Accessing from modules

For multi-function modules, inspect per-function:

```python
module.my_func.inspect(x)
```

---

## Phase 2: MLIR-Level Debug Probes

### The problem

Phase 1's `inspect()` evaluates the computation graph in **cleartext Python** — no encryption, no noise, no MLIR. It's great for checking logic, but it tells you nothing about what actually happens during FHE execution. If simulation introduces a discrepancy (encoding bugs, precision loss, unexpected behavior in the compiled pipeline), `inspect()` won't catch it.

### The solution: `circuit.run_with_probes()`

`run_with_probes()` compiles and runs your circuit in **simulation mode** with debug probe ops injected into the MLIR pipeline. During execution, each probe captures the decoded plaintext value at that point and writes it to a shared buffer. After execution, the buffer is read back into Python as `ProbeSnapshot` objects.

```python
from concrete import fhe

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

### Comparing probes with cleartext inspection

You can compare simulation probe values against the cleartext inspector to verify they agree:

```python
inspection = circuit.inspect(5)
probed = circuit.run_with_probes(5)

print(probed.compare_with(inspection))
```

Output:

```
Node                            Inspect (cleartext)    Probe (simulation)  Match
----------------------------------------------------------------------------------------------------
input                                             5                     -  (no probe)
add                                               6                     6  OK
multiply                                         12                    12  OK
```

If simulation and cleartext disagree, the `Match` column shows `MISMATCH`, pointing you directly to where the pipeline diverges.

### Choosing which nodes to probe

By default, `run_with_probes()` probes all encrypted (non-input) nodes. You can narrow the set with the `probes` parameter:

**Probe by tag:**

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    with fhe.tag("step1"):
        y = x + 1
    with fhe.tag("step2"):
        z = y * 2
    return z

circuit = f.compile(range(8))

# Only probe nodes tagged "step1"
probed = circuit.run_with_probes(5, probes=["step1"])
print(len(probed))  # 1
```

**Probe by predicate:**

```python
# Probe only nodes whose operation involves a table lookup
probed = circuit.run_with_probes(5, probes=lambda node: node.converted_to_table_lookup)
```

**Probe everything (default):**

```python
probed = circuit.run_with_probes(5)  # probes=None → all encrypted nodes
```

If the probe spec matches nothing, the circuit runs normally and returns an empty `ProbeResult`.

### Filtering probe results

The same filtering API from Phase 1 is available on `ProbeResult`:

```python
probed = circuit.run_with_probes(5)

# By operation name
probed.filter(operation_filter="add")

# By tag
probed.filter(tag_filter="step1")

# By encryption status
probed.filter(is_encrypted_filter=True)

# Overflows only
probed.filter(overflow_only=True)

# Custom predicate
probed.filter(custom_filter=lambda snap: snap.value > 10)
```

### Overflow detection

Probes detect overflows the same way `inspect()` does — by checking the captured value against the node's output dtype range:

```python
probed = circuit.run_with_probes(100)
print(probed.has_overflow)  # True

for snap in probed.overflows:
    print(snap)
```

### Browsing individual probe snapshots

Each `ProbeSnapshot` carries:

| Property | Description |
|----------|-------------|
| `snap.value` | Decoded plaintext value captured during simulation |
| `snap.probe_id` | Unique identifier for this probe |
| `snap.node` | The computation graph `Node` this probe is attached to |
| `snap.node_tag` | User-assigned tag (from `fhe.tag(...)`) |
| `snap.tag` | Tag string stored in the MLIR probe op |
| `snap.operation_name` | `"add"`, `"multiply"`, `"tlu"`, etc. |
| `snap.location` | Source file and line number |
| `snap.is_encrypted` | Whether the node output is encrypted |
| `snap.overflow` | Whether the value exceeds the node's dtype range |

Iterate all snapshots:

```python
for snap in probed:
    print(snap.probe_id, snap.operation_name, snap.value)
```

### How it works

Each call to `run_with_probes()`:

1. Resolves the probe spec to a set of graph nodes
2. Recompiles the MLIR with `Tracing.debug_probe` ops inserted after each probed node
3. Resets the global `ProbeBuffer`
4. Runs the circuit through the simulation pipeline (SimulateTFHE → TracingToCAPI → runtime)
5. The runtime `memref_debug_probe_plaintext()` function writes `(probe_id, value, tag)` tuples into the buffer
6. Python reads the buffer back and maps each entry to its originating graph node

Because each call recompiles, there is a compilation cost. For repeated probing with the same probe set, consider caching the result or probing once with a broad spec.

### Important notes

- **Simulation mode only.** Probes capture decoded plaintext values after the SimulateTFHE pass. Encrypted-mode (raw ciphertext capture) is planned for a future phase.
- **Input nodes are not probed.** They are function arguments with known values — there is nothing to capture.
- **Recompilation cost.** Each `run_with_probes()` call recompiles the MLIR. This is fast for small circuits but may be noticeable for large ones.
- **Thread safety.** The probe buffer uses a mutex for concurrent writes. Entries may arrive out of order for parallel circuits; `probe_id` allows correct reconstruction.
- **Large circuits.** A warning is emitted if more than 100 probes are active, since each probe adds memory and execution overhead.

### API reference

**`Circuit.run_with_probes(*args, probes=None)`** / **`FheFunction.run_with_probes(*args, probes=None)`**

Run the circuit in simulation mode with debug probes and return a `ProbeResult`.

- `*args` — input values (same as you'd pass to `simulate()`)
- `probes` — optional probe specification:
  - `None` — probe all encrypted non-input nodes (default)
  - `list[str]` — probe nodes whose tag matches any string in the list
  - `Callable[[Node], bool]` — probe nodes where the predicate returns `True`

**`ProbeResult`**

| Method / Property | Returns |
|---|---|
| `probed.output` | Circuit output from the probed simulation run |
| `len(probed)` | Number of captured probes |
| `probed[i]` | `ProbeSnapshot` at index `i` |
| `for snap in probed` | Iterate all probe snapshots |
| `probed.has_overflow` | `True` if any probe overflowed |
| `probed.overflows` | List of `ProbeSnapshot` with overflow |
| `probed.filter(...)` | Query snapshots by tag, operation, location, encryption, overflow, or custom predicate |
| `probed.summary()` | Formatted table string with probe IDs, operations, values, and overflow markers |
| `probed.compare_with(inspection)` | Side-by-side comparison with an `InspectionResult` from `circuit.inspect()` |

### Accessing from modules

For multi-function modules, probe per-function:

```python
module.my_func.run_with_probes(x, probes=["my_tag"])
```

---

*Future phases will add features below this line.*
