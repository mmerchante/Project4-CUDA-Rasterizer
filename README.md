CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Mariano Merchante
* Tested on
  * Microsoft Windows 10 Pro
  * Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 2601 Mhz, 4 Core(s), 8 Logical Processor(s)
  * 32.0 GB RAM
  * NVIDIA GeForce GTX 1070 (mobile version)
  
## Details
This project implements a hierarchical tiled rasterizer in CUDA. It subdivides the screen multiple times and stores polygon data at different levels to optimize memory usage, and then traverses through this hierarchy when rendering.

## Rasterization
The rasterization aspect is very similar to other approaches, with the difference that instead of iterating through all primitives and doing scanline rasterization, it follows these steps:

- Builds a tile data structure that contains all hierarchy levels and enough memory for primitive indices.
  - It uses a logarithmic scale to increase the primitive capacity as the tile becomes bigger.
- On each frame:
  - Clears every tile primitive counter
  - Iterates over all primitives and stores them at the correct level on the hierarchy. It uses an atomic counter to keep track of how many primitives the tile must render.
  - Rasterizes each tile, iterating through all found primitives and up through the hierarchy until the biggest level is reached.
    - Note that because each tile is running in parallel, z testing has no race condition and thus can be trivially done in the tile kernel.
    
## Specific optimizations

- Backface culling
- Early Z-reject 
- Pineda algorithm for triangle rasterization
   
## Results
This approach seems to be very good when the geometry is balanced throughout different tile levels. If, for example, the full scene can be stored on one small tile, performance can drop dramatically, and can even lose primitives. This can be mitigated by doing multiple passes until all geometry is rasterized, but it is not implemented.

Memory consumption is a big issue too, and the logarithmic scale used for different hierarchy capacities is used to mitigate the fact that as tiles become bigger, more primitives are going to intersect with them. 




### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
