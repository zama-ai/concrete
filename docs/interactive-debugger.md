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

*Future phases will add features below this line.*
