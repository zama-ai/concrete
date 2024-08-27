# Formatting and drawing
This document explains how to format and draw a compiled circuit in Python. 

## Formatting

To convert your compiled circuit into its textual representation, use the `str` function:

```python
str(circuit)
```

If you just want to see the output on your terminal, you can directly print it as well:

```python
print(circuit)
```

{% hint style="warning" %}
Formatting is designed for debugging purpose only. It's not possible to create the circuit back from its textual representation. See [How to Deploy](../guides/deploy.md) if that's your goal.
{% endhint %}

## Drawing

{% hint style="danger" %}
Drawing functionality requires the installation of the package with the full feature set. See the [Installation](../get-started/installing.md) section for instructions.
{% endhint %}

To draw your compiled circuit, use the `draw` method:

```python
drawing = circuit.draw()
```

This method draws the circuit, saves it as a temporary PNG file and returns the file path.

You can display the drawing in a Jupyter notebook:

```python
from PIL import Image
drawing = Image.open(circuit.draw())
drawing.show()
drawing.close()
```

Alternatively, you can use the `show` option of the `draw` method to display the drawing with `matplotlib`:

```python
circuit.draw(show=True)
```

{% hint style="danger" %}
Using this option will clear any existing matplotlib plots.
{% endhint %}

Lastly, to save the drawing to a specific path, use the `save_to` option:

```python
destination = "/tmp/path/of/your/choice.png"
drawing = circuit.draw(save_to=destination)
assert drawing == destination
```
