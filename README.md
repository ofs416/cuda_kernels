# cuda_kernels

In this repository I will brush up on c, dive deeper into low-level programming and optimisations to aide scientific computing.
Most scientific programming I do involves large tensors, hence understanding the concepts required to effectively use parallel programming and efficient memory access will increase overall workflow efficiency.

## Objectives

- Familiarise myself with c: syntax, pointers, structs, memory management (basically things not present in python which I'm most familiar with)
- Understand low-level conceptes, such as caching, in-order to optimise code (such as row-major vs column-major matrices)
- Learn cuda c, implementing my own matmul kernel with different levels of optimisation (tiling and more)
- Create a perceptron using CuBlas and it's ability to merge kernels
- Create kernels that can be accessed with pytorch
- Explore higher-level abstractions of cuda such as Triton - used by pytorch - and Taichi
- Brush up on 2nd year Fluid mechanics and create an simulated air tunnel flow over a 2D airfoil with Taichi

