# Triangle-Rasterize-Numpy

A script to rasterize a triangle using numpy, interpolating attributes attached to vertices.

## Usage

```python
import numpy as np
from triangle import draw_triangle

attrib = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]).astype(np.float32)

coords = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.5]
    ]).astype(np.float32)

img = np.zeros((512, 512, attrib.shape[1]))

draw_triangle(img, coords, attrib)
```
