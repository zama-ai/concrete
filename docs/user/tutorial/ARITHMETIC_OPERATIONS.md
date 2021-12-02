```{warning}
FIXME(Umut): update a bit, with the new API
FIXME(Umut/Arthur): update a bit to explain things with the tensors and new operations. At the same time, I think we can exhaustively give examples of every supported functions, since we start to have a lot, so maybe, we would just explain a bit?
FIXME(all): actually, I am not even sure we should keep this .md, it can't be exhaustive enough, and looks pretty trivial. What do you think

```

# Arithmetic Operations

In this tutorial, we are going to go over all arithmetic operations available in **Concrete**. Please read [Compiling and Executing](../howto/COMPILING_AND_EXECUTING.md) before reading further to see how you can compile the functions below.

## Addition

### Static ClearScalar and EncryptedScalar

<!--python-test:skip-->
```python
def f(x):
    return x + 42
```

or

<!--python-test:skip-->
```python
def f(x):
    return 42 + x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run(3) == 45
circuit.run(0) == 42
```

### Dynamic ClearScalar and EncryptedScalar

<!--python-test:skip-->
```python
def f(x, y):
    return x + y
```

or

<!--python-test:skip-->
```python
def f(x, y):
    return y + x
```

results in

<!--python-test:skip-->
```python
circuit.run(6, 4) == 10
circuit.run(1, 1) == 2
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = ClearScalar(UnsignedInteger(bits))`

### EncryptedScalar and EncryptedScalar

<!--python-test:skip-->
```python
def f(x, y):
    return x + y
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = EncryptedScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run(7, 7) == 14
circuit.run(3, 4) == 7
```

## Subtraction

### Static ClearScalar and EncryptedScalar 

<!--python-test:skip-->
```python
def f(x):
    return 3 - x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run(2) == 1
circuit.run(3) == 0
```

### Dynamic ClearScalar and EncryptedScalar

<!--python-test:skip-->
```python
def f(x, y):
    return y - x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = ClearScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run(2, 4) == 2
circuit.run(1, 7) == 6
```

## Multiplication

### Static ClearScalar and EncryptedScalar

<!--python-test:skip-->
```python
def f(x):
    return x * 2
```

or

<!--python-test:skip-->
```python
def f(x):
    return 2 * x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run(2) == 4
circuit.run(5) == 10
```

### Dynamic ClearScalar and EncryptedScalar

<!--python-test:skip-->
```python
def f(x, y):
    return x * y
```

or

<!--python-test:skip-->
```python
def f(x, y):
    return y * x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = ClearScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run(2, 3) == 6
circuit.run(1, 7) == 7
```

## Dot Product

### Dynamic ClearTensor and EncryptedTensor

<!--python-test:skip-->
```python
def f(x, y):
    return np.dot(x, y)
```

or

<!--python-test:skip-->
```python
def f(x, y):
    return np.dot(y, x)
```

where

- `x = EncryptedTensor(UnsignedInteger(bits), shape=(2,))`
- `y = ClearTensor(UnsignedInteger(bits), shape=(2,))`

results in

<!--python-test:skip-->
```python
circuit.run([1, 1], [2, 3]) == 5
circuit.run([2, 3], [2, 3]) == 13
```

## Combining all together

<!--python-test:skip-->
```python
def f(x, y, z):
    return 100 - (2 * (np.dot(x, y) + z))
```

where

- `x = EncryptedTensor(UnsignedInteger(bits), shape=(2,))`
- `y = ClearTensor(UnsignedInteger(bits), shape=(2,))`
- `z = EncryptedScalar(UnsignedInteger(bits))`

results in

<!--python-test:skip-->
```python
circuit.run([1, 2], [4, 3], 10) == 60
circuit.run([2, 3], [3, 2], 5) == 66
```
