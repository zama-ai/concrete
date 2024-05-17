# Performance optimisation

This document collects common examples to illustrate how to optimize code for better performance under Fully Homomorphic Encryption (FHE) constraints. (Learn more about these constraints in [Basic of FHE](fhe\_basics.md).)

## Minimum for two values

In this example, we compute the minimum of two numbers `y` and `x`,  by computing the difference between two numbers and conditionally subtracting this difference from `y`. This gives either `x` if `y>x` ,or `y` if `x>y`:

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def min_two(x, y):
	diff = y - x
	min_x_y = y - np.maximum(y - x, 0)
	return min_x_y

inputset = [tuple(np.random.randint(0, 16, size=2)) for _ in range(50)]
circuit = min_two.compile(inputset)

x, y = np.random.randint(0, 16, size=2)
assert circuit.encrypt_run_decrypt(x, y) == min(x, y)
```

## Maximum for two values

The following example computes the maximum of two integers with the same method as the previous example.

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def max_two(x, y):
	diff = y - x
	max_x_y = y - np.minimum(y - x, 0)
	return max_x_y

inputset = [tuple(np.random.randint(0, 16, size=2)) for _ in range(50)]
circuit = max_two.compile(inputset)

x, y = np.random.randint(0, 16, size=2)
assert circuit.encrypt_run_decrypt(x, y) == max(x, y)
```

## Minimum for several values

Here's an extension for more than two values:

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"args": "encrypted"})
def fhe_min(args):
    remaining = list(args)
    while len(remaining) > 1:
        a = remaining.pop()
        b = remaining.pop()
        min_a_b = b - np.maximum(b - a, 0)
        remaining.insert(0, min_a_b)
    return remaining[0]

inputset = [np.random.randint(0, 16, size=5) for _ in range(50)]
circuit = fhe_min.compile(inputset)

x1, x2, x3, x4, x5 = np.random.randint(0, 16, size=5)
assert circuit.encrypt_run_decrypt([x1, x2, x3, x4, x5]) == min(x1, x2, x3, x4, x5)
```

## Retrieving a value within an encrypted array with an encrypted index

This example shows how to handle an array with an encrypted index. It creates a "selection" array filled with `0` s except for the requested index that will be set to `1`, and sum the products of all array values by this selection array:

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"array": "encrypted", "index": "encrypted"})
def indexed_value(array, index):
    all_indices = np.arange(array.size)
    index_selection = index == all_indices
    selection_and_zeros = array * index_selection
    selection = np.sum(selection_and_zeros)
    return selection

inputset = [(np.random.randint(0, 16, size=5), np.random.randint(0, 5)) for _ in range(50)]
circuit = indexed_value.compile(inputset)

array = np.random.randint(0, 16, size=5)

index = np.random.randint(0, 5)
assert circuit.encrypt_run_decrypt(array, index) == array[index]
```

## Filter an array with comparison (>)

This example filters an encrypted array by an encrypted condition -`greater than` with an encrypted value. It packs all values with a selection bit resulting from the comparison, which allows to only unpack the filtered values:

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"numbers": "encrypted", "threshold": "encrypted"})
def filtering(numbers, threshold):
    is_greater = numbers > threshold

    shifted_numbers = numbers * 2  # open space for a single bit at the end
    combined_numbers_and_is_greater = shifted_numbers + is_greater  # put is_greater to that bit

    def extract(combination):
        is_greater = (combination % 2) == 1  # extract is_greater back from packing
        if_true = combination // 2  # if is greater is true, we unpack the number and use it
        if_false = 0  # otherwise we set the element to zero
        return np.where(is_greater, if_true, if_false)  # and apply the operation

    return fhe.univariate(extract)(combined_numbers_and_is_greater)

inputset = [(np.random.randint(0, 16, size=5), np.random.randint(0, 16)) for _ in range(50)]
circuit = filtering.compile(inputset)

numbers = np.random.randint(0, 16, size=5)
threshold = np.random.randint(0, 16)
assert np.array_equal(circuit.encrypt_run_decrypt(numbers, threshold), list(map(lambda x: x if x > threshold else 0, numbers)))

```

## Matrix Row/Col means

In this example matrix operation, we split the values into sub-groups and calculate the mean of the sub-groups instead of sequentially summing all values to calculate the mean value.

This approach applies one of the key optimization concepts in **Concrete -** parallelization.

```python
import numpy as np
from concrete import fhe

def smallest_prime_divisor(n):
    if n % 2 == 0:
        return 2

    for i in range(3, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return i

    return n

def mean_of_vector(x):
    assert x.size != 0
    if x.size == 1:
        return x[0]

    group_size = smallest_prime_divisor(x.size)
    if x.size == group_size:
        return np.round(np.sum(x) / x.size).astype(np.int64)

    groups = []
    for i in range(x.size // group_size):
        start = i * group_size
        end = start + group_size
        groups.append(x[start:end])

    mean_of_groups = []
    for group in groups:
        mean_of_groups.append(np.round(np.sum(group) / group_size).astype(np.int64))

    return mean_of_vector(fhe.array(mean_of_groups))

@fhe.compiler(({"x": "encrypted"}))
def mean_of_matrix(x):
    return mean_of_vector(x.flatten())

@fhe.compiler(({"x": "encrypted"}))
def mean_of_rows_of_matrix(x):
    means = []
    for i in range(x.shape[0]):
        means.append(mean_of_vector(x[i]))
    return fhe.array(means)

@fhe.compiler(({"x": "encrypted"}))
def mean_of_columns_of_matrix(x):
    means = []
    for i in range(x.shape[1]):
        means.append(mean_of_vector(x[:, i]))
    return fhe.array(means)


inputset = [np.random.randint(0, 16, size=(5,5)) for _ in range(50)]
matrix = np.random.randint(0, 16, size=(5, 5))

circuit = mean_of_matrix.compile(inputset)
assert circuit.encrypt_run_decrypt(matrix) == round(matrix.mean())

circuit = mean_of_rows_of_matrix.compile(inputset)
assert np.array_equal(circuit.encrypt_run_decrypt(matrix), [round(x) for x in matrix.mean(1)])

circuit = mean_of_columns_of_matrix.compile(inputset)
assert np.array_equal(circuit.encrypt_run_decrypt(matrix), [round(x) for x in matrix.mean(0)])
```

{% hint style="info" %}
All code snippets in this document are temporary workarounds sourced from our [community forum](https://community.zama.ai) support answers.  Future versions of Concrete may include these functions in a more generic and efficient form.
{% endhint %}
