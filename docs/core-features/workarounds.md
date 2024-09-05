# Common tricks

This document introduces several common techniques for optimizing code to fit Fully Homomorphic Encryption (FHE) [constraints](../core-features/fhe_basics.md#pbs-management). The examples provided demonstrate various workarounds and performance optimizations that you can implement while working with the **Concrete** library.

{% hint style="info" %}
All code snippets provided here are temporary workarounds. In future versions of Concrete, some functions described here could be directly available in a more generic and efficient form. These code snippets are coming from support answers in our [community forum](https://community.zama.ai)
{% endhint %}

## Retrieving a value within an encrypted array with an encrypted index
This example demonstrates how to retrieve a value from an array using an encrypted index. The method creates a "selection" array filled with `0`s except for the requested index, which will be `1`. It then sums the products of all array values with this selection array:

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

This example filters an encrypted array with an encrypted condition, in this case a `greater than` comparison with an encrypted value. It packs all values with a selection bit that results from the comparison, allowing the unpacking of only the filtered values:

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
This example introduces a key concept when using **Concrete**: maximizing parallelization. Instead of sequentially summing all values to compute a mean, the values are split into sub-groups, and the mean of these sub-group means is computed:

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
