'''
MIT License

Copyright (c) 2023 LuniumLuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

#
# Draw a triangle with vertex attributes interpolated
# based on barycentric coordinates with Numpy
#

import numpy as np

# img: image to draw on, must be ndarray in (H, W, C)
# coords: 2D triangle vertex coordinates in (3, 2)
# attrib: vertex attributes to interpolate in (3, C)
# eps: bigger epsilon enable more pixels around border to be drawn
def draw_triangle(img        : np.ndarray, 
                  coords     : np.ndarray,
                  attrib     : np.ndarray,
                  eps        = 1e-6) -> None:
                  
    assert isinstance(img,    np.ndarray)
    assert isinstance(coords, np.ndarray)
    assert isinstance(attrib, np.ndarray)
    assert coords.ndim      == 2
    assert coords.shape     == (3, 2)
    assert attrib.ndim      == 2
    assert attrib.shape[0]  == 3
    assert attrib.shape[1]  == img.shape[2]

    resolution = np.array(img.shape[:2]).astype(np.int32)

    f_min = np.min(coords, axis=0)
    f_max = np.max(coords, axis=0)
    d_min = np.clip(np.floor(f_min * resolution).astype(np.int32) - 1, 0, resolution - 1)
    d_max = np.clip(np.ceil(f_max * resolution).astype(np.int32) + 1, 0, resolution - 1)

    x = np.linspace(f_min[0], f_max[0], (d_max - d_min)[0]).astype(np.float32)
    y = np.linspace(f_min[1], f_max[1], (d_max - d_min)[1]).astype(np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing='xy')

    e01  = coords[1] - coords[0]
    e12  = coords[2] - coords[1]
    area = np.cross(e01, e12)

    v = (e01[0] * (grid_y - coords[0,1]) - e01[1] * (grid_x - coords[0,0])) / area
    w = (e12[0] * (grid_y - coords[1,1]) - e12[1] * (grid_x - coords[1,0])) / area
    u = 1.0 - w - v

    interp = attrib[0] * w[:, :, None] + attrib[1] * u[:, :, None] + attrib[2] * v[:, :, None]

    mask_w = np.bitwise_and(w > -eps, w < 1 + eps)
    mask_u = np.bitwise_and(u > -eps, u < 1 + eps)
    mask_v = np.bitwise_and(v > -eps, v < 1 + eps)

    mask = np.bitwise_and(mask_w, mask_u)
    mask = np.bitwise_and(mask,   mask_v)

    img[d_min[0]:d_max[0],d_min[1]:d_max[1]][mask] = 0
    interp[np.bitwise_not(mask)] = 0

    img[d_min[0]:d_max[0],d_min[1]:d_max[1]] += interp


if __name__ == '__main__':
    import time
    from matplotlib import pyplot as plt
    resolution = (720, 1280)

    attrib = np.array([[1], [0.5], [1]]).astype(np.float32)
    coords = []
    N = 10
    for i in range(N):
        for j in range(N):
            x = i / N
            y = j / N
            coords.append([[x, y], [x + 1/N, y + 1/N], [x + 1/N, y]])
            coords.append([[x, y], [x + 1/N, y + 1/N], [x, y + 1/N]])
    coords = np.array(coords).astype(np.float32)
    coords = np.clip(coords, 0.0, 1.0)

    img = np.zeros((resolution[0], resolution[1], attrib.shape[1]))

    start = time.time()

    for i in range(coords.shape[0]):
        draw_triangle(img, coords[i], attrib)

    end = time.time()
    print(f'Time: {end - start}s')

    img = np.clip(img, 0.0, 1.0)
    plt.imshow(img), plt.show()
