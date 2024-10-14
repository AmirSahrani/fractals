#include <cuComplex.h>


__device__ cuDoubleComplex cpow(cuDoubleComplex z, int power) {
    cuDoubleComplex result = make_cuDoubleComplex(1.0, 0.0);
    for (int i = 0; i < power; i++) {
        result = cuCmul(result, z);
    }
    return result;
}

__global__ void iterate_mandelbrot(cuDoubleComplex* c, int* iterations, int width, int height, int max_iterations, int power)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;
    int index = idy * width + idx;

    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    int iter;

    for (iter = 0; iter < max_iterations; iter++) {
        if (cuCabs(z) > 2.0) break;
        
        z = cuCadd(cpow(z, power), c[index]);
    }

    iterations[index] = iter;
}


