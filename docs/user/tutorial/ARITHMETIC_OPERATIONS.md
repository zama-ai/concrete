# Arithmetic Operations

In this tutorial, we are going to go over all arithmetic operations available in **concrete**. Please read [Compiling and Executing](../howto/COMPILING_AND_EXECUTING.md) before reading further to see how you can compile the functions below.

## Addition

### Static ClearScalar and EncryptedScalar

```python
def f(x):
    return x + 42
```

or

```python
def f(x):
    return 42 + x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

```python
engine.run(3) == 45
engine.run(0) == 42
```

### Dynamic ClearScalar and EncryptedScalar

```python
def f(x, y):
    return x + y
```

or

```python
def f(x, y):
    return y + x
```

results in

```python
engine.run(6, 4) == 10
engine.run(1, 1) == 2
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = ClearScalar(UnsignedInteger(bits))`

### EncryptedScalar and EncryptedScalar

```python
def f(x, y):
    return x + y
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = EncryptedScalar(UnsignedInteger(bits))`

results in

```python
engine.run(7, 7) == 14
engine.run(3, 4) == 7
```

## Subtraction

### Static ClearScalar and EncryptedScalar 

```python
def f(x):
    return 3 - x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

```python
engine.run(2) == 1
engine.run(3) == 0
```

### Dynamic ClearScalar and EncryptedScalar

```python
def f(x, y):
    return y - x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = ClearScalar(UnsignedInteger(bits))`

results in

```python
engine.run(2, 4) == 2
engine.run(1, 7) == 6
```

## Multiplication

### Static ClearScalar and EncryptedScalar

```python
def f(x):
    return x * 2
```

or

```python
def f(x):
    return 2 * x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

```python
engine.run(2) == 4
engine.run(5) == 10
```

### Dynamic ClearScalar and EncryptedScalar

```python
def f(x, y):
    return x * y
```

or

```python
def f(x, y):
    return y * x
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`
- `y = ClearScalar(UnsignedInteger(bits))`

results in

```python
engine.run(2, 3) == 6
engine.run(1, 7) == 7
```

## Dot Product

### Dynamic ClearTensor and EncryptedTensor

```python
def f(x, y):
    return np.dot(x, y)
```

or

```python
def f(x, y):
    return np.dot(y, x)
```

where

- `x = EncryptedTensor(UnsignedInteger(bits), shape=(2,))`
- `y = ClearTensor(UnsignedInteger(bits), shape=(2,))`

results in

```python
engine.run([1, 1], [2, 3]) == 5
engine.run([2, 3], [2, 3]) == 13
```

## Combining all together

```python
def f(x, y, z):
    return 100 - (2 * (np.dot(x, y) + z))
```

where

- `x = EncryptedTensor(UnsignedInteger(bits), shape=(2,))`
- `y = ClearTensor(UnsignedInteger(bits), shape=(2,))`
- `z = EncryptedScalar(UnsignedInteger(bits))`

results in

```python
engine.run([1, 2], [4, 3], 10) == 60
engine.run([2, 3], [3, 2], 5) == 66
```
