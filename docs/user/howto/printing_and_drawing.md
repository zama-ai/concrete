# Printing and Drawing a FHE circuit

Sometimes, it can be useful to print or draw fhe circuits, we provide methods to just do that. Please read [Compiling and Executing](../basics/compiling_and_executing.md) before reading further to see how you can compile your function into an fhe circuit.

## Printing

To print your circuit, you can do the following:

<!--pytest-codeblocks:skip-->
```python
print(circuit)
```

## Drawing

```{WARNING}
The draw function requires the installation of the package's extra dependencies.

The drawing package required is `pygraphviz` which needs `graphviz` packages installed on your OS, see <a href="https://pygraphviz.github.io/documentation/stable/install.html">https://pygraphviz.github.io/documentation/stable/install.html</a>

To install the required drawing packages once you have `graphviz` installed run:

`pip install concretefhe[full]`

You may need to force reinstallation

`pip install --force-reinstall concretefhe[full]`
```

To draw your circuit, you can do the following:

<!--pytest-codeblocks:skip-->
```python
drawing = circuit.draw()
```

This method will draw the circuit on a temporary PNG file and return the path to this file.

To show the drawing, you can use the following code in a jupyter notebook.

<!--pytest-codeblocks:skip-->
```python
from PIL import Image
drawing = Image.open(circuit.draw())
drawing.show()
drawing.close()
```

Additionally, you can use the `show` option of the `draw` method to show the drawing with matplotlib. Beware that this will clear the matplotlib plots you have.

<!--pytest-codeblocks:skip-->
```python
circuit.draw(show=True)
```

Lastly, you can save the drawing to a specific path like this:

<!--pytest-codeblocks:skip-->
```python
destination = "/tmp/path/of/your/choice.png"
drawing = circuit.draw(save_to=destination)
assert drawing == destination
```
