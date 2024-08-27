# Progressbar
This document introduces the progressbar feature that provides visual feedback on the execution progress of large circuits, which can take considerable time to execute. 

The following Python code demonstrates how to enable and use the progressbar:

```python
import time

import matplotlib.pyplot as plt
import numpy as np
import randimage
from concrete import fhe

configuration = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",

    # To enable displaying progressbar
    show_progress=True,
    # To enable showing tags in the progressbar (does not work in notebooks)
    progress_tag=True,
    # To give a title to the progressbar
    progress_title="Evaluation:",
)

@fhe.compiler({"image": "encrypted"})
def to_grayscale(image):
    with fhe.tag("scaling.r"):
        r = image[:, :, 0]
        r = (r * 0.30).astype(np.int64)

    with fhe.tag("scaling.g"):
        g = image[:, :, 1]
        g = (g * 0.59).astype(np.int64)

    with fhe.tag("scaling.b"):
        b = image[:, :, 2]
        b = (b * 0.11).astype(np.int64)

    with fhe.tag("combining.rgb"):
        gray = r + g + b
        
    with fhe.tag("creating.result"):
        gray = np.expand_dims(gray, axis=2)
        result = np.concatenate((gray, gray, gray), axis=2)
    
    return result

image_size = (16, 16)
image_data = (randimage.get_random_image(image_size) * 255).round().astype(np.int64)

print()

print(f"Compilation started @ {time.strftime('%H:%M:%S', time.localtime())}")
start = time.time()
inputset = [np.random.randint(0, 256, size=image_data.shape) for _ in range(100)]
circuit = to_grayscale.compile(inputset, configuration)
end = time.time()
print(f"(took {end - start:.3f} seconds)")

print()

print(f"Key generation started @ {time.strftime('%H:%M:%S', time.localtime())}")
start = time.time()
circuit.keygen()
end = time.time()
print(f"(took {end - start:.3f} seconds)")

print()

print(f"Evaluation started @ {time.strftime('%H:%M:%S', time.localtime())}")
start = time.time()
grayscale_image_data = circuit.encrypt_run_decrypt(image_data)
end = time.time()
print(f"(took {end - start:.3f} seconds)")

fig, axs = plt.subplots(1, 2)
axs = axs.flatten()

axs[0].set_title("Original")
axs[0].imshow(image_data)
axs[0].axis("off")

axs[1].set_title("Grayscale")
axs[1].imshow(grayscale_image_data)
axs[1].axis("off")

plt.show()
```

When you run this code, you will see a progressbar like this one:
```
Evaluation:  10% |█████.............................................|  10% (scaling.r)
^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^
Title        Progressbar                                                   Tag
```

As the execution proceeds, the progress bar updates:
```
Evaluation:  30% |███████████████...................................|  30% (scaling.g)
```
```
Evaluation:  50% |█████████████████████████.........................|  50% (scaling.b)
```

{% hint style="info" %}
The progress bar does not measure time. When it shows 50%, it indicates that half of the nodes in the computation graph have been processed, not that half of the time has elapsed. The duration of processing different node types may vary, so the progress bar should not be used to estimate the remaining time.
{% endhint %}

Once the progressbar fills and execution completes, you will see the following figure:

![](../\_static/progress/grayscale.png)
