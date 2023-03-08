# Compilation

The compilation journey begins with tracing to get an easy-to-manipulate representation of the function. We call this representation a `Computation Graph`, which is basically a Directed Acyclic Graph (DAG) containing nodes representing the computations done in the function. Working with graphs is good because they have been studied extensively over the years and there are a lot of algorithms to manipulate them. Internally, we use [networkx](https://networkx.org), which is an excellent graph library for Python.

The next step in the compilation is transforming the computation graph. There are many transformations we perform, and they will be discussed in their own sections. In any case, the result of transformations is just another computation graph.

After transformations are applied, we need to determine the bounds (i.e., the minimum and the maximum values) of each intermediate node. This is required because FHE currently allows a limited precision for computations. Bound measurement is our way to know what is the required precision for the function.

The final step is to transform the computation graph to equivalent `MLIR` code. How this is done will be explained in detail in its own chapter.

Once the MLIR is generated, we send it to the **Concrete-Compiler**, and it completes the compilation process.

## Tracing

Given a Python function `f` such as this one:

```
def f(x):
    return (2 * x) + 3
```

...the goal of tracing is to create the following computation graph without needing any change from the user.

![](../\_static/compilation-pipeline/two\_x\_plus\_three.png)

(Note that the edge labels are for non-commutative operations. To give an example, a subtraction node represents `(predecessor with edge label 0) - (predecessor with edge label 1)`)

To do this, we make use of `Tracer`s, which are objects that record the operation performed during their creation. We create a `Tracer` for each argument of the function and call the function with those tracers. `Tracer`s make use of the operator overloading feature of Python to achieve their goal:

```
def f(x, y):
    return x + 2 * y

x = Tracer(computation=Input("x"))
y = Tracer(computation=Input("y"))

resulting_tracer = f(x, y)
```

`2 * y` will be performed first, and `*` is overloaded for `Tracer` to return another tracer: `Tracer(computation=Multiply(Constant(2), self.computation))`, which is equal to `Tracer(computation=Multiply(Constant(2), Input("y")))`

`x + (2 * y)` will be performed next, and `+` is overloaded for `Tracer` to return another tracer: `Tracer(computation=Add(self.computation, (2 * y).computation))`, which is equal to `Tracer(computation=Add(Input("x"), Multiply(Constant(2), Input("y")))`

In the end, we will have output tracers that can be used to create the computation graph. The implementation is a bit more complex than this, but the idea is the same.

Tracing is also responsible for indicating whether the values in the node would be encrypted or not, and the rule for that is if a node has an encrypted predecessor, it is encrypted as well.

## Topological transforms

The goal of topological transforms is to make more functions compilable.

With the current version of **Concrete-Numpy**, floating-point inputs and floating-point outputs are not supported. However, if the floating-point operations are intermediate operations, they can sometimes be fused into a single table lookup from integer to integer, thanks to some specific transforms.

Let's take a closer look at the transforms we can currently perform.

### Fusing.

We have allocated a whole new chapter to explaining fusing. You can find it after this chapter.

## Bounds measurement

Given a computation graph, the goal of the bound measurement step is to assign the minimal data type to each node in the graph.

Let's say we have an encrypted input that is always between `0` and `10`. We should assign the type `Encrypted<uint4>` to the node of this input as `Encrypted<uint4>` is the minimal encrypted integer that supports all values between `0` and `10`.

If there were negative values in the range, we could have used `intX` instead of `uintX`.

Bounds measurement is necessary because FHE supports limited precision, and we don't want unexpected behaviour while evaluating the compiled functions.

Let's take a closer look at how we perform bounds measurement.

### Inputset evaluation.

This is a simple approach that requires an inputset to be provided by the user.

The inputset is not to be confused with the dataset, which is classical in ML, as it doesn't require labels. Rather, it is a set of values which are typical inputs of the function.

The idea is to evaluate each input in the inputset and record the result of each operation in the computation graph. Then we compare the evaluation results with the current minimum/maximum values of each node and update the minimum/maximum accordingly. After the entire inputset is evaluated, we assign a data type to each node using the minimum and the maximum values it contains.

Here is an example, given this computation graph where `x` is encrypted:

![](../\_static/compilation-pipeline/two\_x\_plus\_three.png)

and this inputset:

```
[2, 3, 1]
```

Evaluation Result of `2`:

* `x`: 2
* `2`: 2
* `*`: 4
* `3`: 3
* `+`: 7

New Bounds:

* `x`: \[**2**, **2**]
* `2`: \[**2**, **2**]
* `*`: \[**4**, **4**]
* `3`: \[**3**, **3**]
* `+`: \[**7**, **7**]

Evaluation Result of `3`:

* `x`: 3
* `2`: 2
* `*`: 6
* `3`: 3
* `+`: 9

New Bounds:

* `x`: \[2, **3**]
* `2`: \[2, 2]
* `*`: \[4, **6**]
* `3`: \[3, 3]
* `+`: \[7, **9**]

Evaluation Result of `1`:

* `x`: 1
* `2`: 2
* `*`: 2
* `3`: 3
* `+`: 5

New Bounds:

* `x`: \[**1**, 3]
* `2`: \[2, 2]
* `*`: \[**2**, 6]
* `3`: \[3, 3]
* `+`: \[**5**, 9]

Assigned Data Types:

* `x`: Encrypted<**uint2**>
* `2`: Clear<**uint2**>
* `*`: Encrypted<**uint3**>
* `3`: Clear<**uint2**>
* `+`: Encrypted<**uint4**>

## MLIR conversion

The actual compilation will be done by the **Concrete-Compiler**, which is expecting an MLIR input. The MLIR conversion goes from a computation graph to its MLIR equivalent. You can read more about it [here](mlir.md).
