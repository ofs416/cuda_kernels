# CUDA Kernels

In this repository, I will brush up on C, dive deeper into low-level programming, and optimizations to aid scientific computing. Most scientific programming I do involves large tensors. Hence, understanding the concepts required to effectively use parallel programming and efficient memory access will increase overall workflow efficiency.

## Objectives

- **Familiarize with C**: 
  - Syntax
  - Pointers
  - Structs
  - Memory management
  - (Basically things not present in Python, which I'm most familiar with)
- **Understand low-level concepts**:
  - Caching
  - Optimizing code (e.g., row-major vs column-major matrices)
- **Learn CUDA C**:
  - Implement my own matrix multiplication kernel with different levels of optimization (e.g., tiling)
- ~~**Create a perceptron using CuBLAS**:~~
  - ~~Leverage its ability to merge kernels~~
- **Create a convolution kernel and use it for classification with MNIST**:
  - Train the model with pytorch
  - Deploy using the kernel ported to python
- **Develop kernels accessible with PyTorch**
- **Explore higher-level CUDA abstractions**:
  - Numba
  - Triton (used by PyTorch)
  - Taichi (a DSL with python frontend)
