# Printing and Drawing

Sometimes, it can be useful to print or draw fhe circuits, we provide methods to just do that. Please read [Compiling and Executing](../howto/COMPILING_AND_EXECUTING.md) before reading further to see how you can compile your function into an fhe circuit.

## Printing

To print your circuit, you can do the following:

<!--python-test:skip-->
```python
print(circuit)
```

## Drawing

To draw your circuit, you can do the following:

<!--python-test:skip-->
```python
drawing = circuit.draw()
```

This method will draw the circuit on a temporary PNG file and return the path to this file.

To show the drawing, you can use the following code in a jupyter notebook.

<!--python-test:skip-->
```python
from PIL import Image
drawing = Image.open(circuit.draw())
drawing.show()
drawing.close()
```

Additionally, you can use the `show` option of the `draw` method to show the drawing with matplotlib. Beware that this will clear the matplotlib plots you have.

<!--python-test:skip-->
```python
circuit.draw(show=True)
```

Lastly, you can save the drawing to a specific path like this:

<!--python-test:skip-->
```python
destination = "/tmp/path/of/your/choice.png"
drawing = circuit.draw(save_to=destination)
assert drawing == destination
```
